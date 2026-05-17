# Docker 文件系统模型与 editable 安装

> 整理自 verl 环境配置过程中的概念梳理。覆盖 image / container / volume / pip install -e 的底层机制。

## 目录

- [Image 与 Container 的关系](#image-与-container-的关系)
- [Union FS 与 Copy-on-Write](#union-fs-与-copy-on-write)
- [Volume / Bind Mount：双向直通](#volume--bind-mount双向直通)
- [pip install -e 的本质](#pip-install--e-的本质)
- [完整链路：宿主机改代码 → 容器立即生效](#完整链路宿主机改代码--容器立即生效)
- [实用对照表](#实用对照表)
- [心智模型一句话总结](#心智模型一句话总结)

---

## Image 与 Container 的关系

### 核心概念

- **Image**：只读模板（类比：光盘 / 类）
- **Container**：image 的运行实例（类比：装入电脑后的 C 盘 / 对象）
- 一个 image 可以同时对应多个 container，每个 container 有独立的可写层

```
image (只读)
  ├── container A (可写层) ← 你 pip install、echo >>、mkdir 都写到这里
  ├── container B (可写层) ← 另一个独立的可写层
  └── container C (可写层)
```

### 镜像的分层结构

镜像由多个**只读层**组成，每一层对应 Dockerfile 里的一条 `RUN` / `COPY` 指令：

```
verlai/verl:vllm017.latest
├── Layer 1 (只读): ubuntu base rootfs     (~100MB)
├── Layer 2 (只读): CUDA 12.9               (~3GB)
├── Layer 3 (只读): cuDNN / PyTorch         (~5GB)
├── Layer 4 (只读): vLLM / flash-attn       (~8GB)
├── Layer 5 (只读): Megatron / TE           (~4GB)
└── Layer 6 (只读): 其他 python 包           (~2GB)
```

**关键**：所有只读层在磁盘上**只存一份**，被所有容器共享。跑 10 个容器不会占 10 倍空间。

---

## Union FS 与 Copy-on-Write

### 容器启动时的文件系统

```
     container (运行中)
     ┌────────────────────────┐
     │ Writable Layer (可写)   │ ← 所有写操作都进这里
     ├────────────────────────┤
     │ Layer 6 (只读)          │
     │ Layer 5 (只读)          │
     │ Layer 4 (只读)          │   ← 镜像的只读层
     │ Layer 3 (只读)          │      所有容器共享
     │ Layer 2 (只读)          │
     │ Layer 1 (只读)          │
     └────────────────────────┘
```

Docker 用 **OverlayFS**（联合文件系统）把所有层合并成一个看似普通的根目录 `/`。
启动容器时，在镜像顶部加一层空的可写层。

### Copy-on-Write 三种情况

**情况 A：创建新文件**（`touch /root/test.txt`）
- 直接写到可写层，只读层无影响

**情况 B：修改已有文件**（`echo xxx >> /root/.bashrc`）
- 原文件在某个只读层
- Docker 先把它**复制一份到可写层**
- 然后修改可写层那份
- 只读层的原版完全没变
- `ls /root/` 看到的是可写层那份（上层覆盖下层）

**情况 C：删除只读层里的文件**（`rm /usr/bin/python3`）
- 真正的 `python3` 在只读层，动不了
- Docker 在可写层放一个**白障文件（whiteout）**标记它被删了
- `ls` 看不到了，但底层文件依然存在
- 删容器后白障消失，`python3` 又回来

### 验证命令

```bash
# 看可写层 vs 总虚拟大小
docker ps -s
# CONTAINER   IMAGE   ...   SIZE
# xxxxxxxx    ...     ...   45MB (virtual 42.4GB)
#                           ↑可写层  ↑全部层

# 看可写层都改了哪些文件
docker diff verl
# A /root/.bashrc                                ← Added
# A /usr/local/lib/python3.12/site-packages/...
# C /tmp                                         ← Changed
# D /old/file                                    ← Deleted
```

### 为什么这么设计

1. **节省空间**：多容器共享镜像层
2. **快速启动**：只需要建空可写层
3. **不可变性**：镜像永远不变，跨环境字节一致
4. **可复现**：删容器即重置

---

## Volume / Bind Mount：双向直通

### 本质：绕开 OverlayFS

`-v ~/tengee_workplace/verl:/workspace/verl` 这条挂载**完全不经过 CoW**：

```
容器视角：

     /
     ├── usr/      ← OverlayFS（只读层）
     ├── root/     ← OverlayFS（只读层 + 可写层）
     ├── workspace/
     │   └── verl/  ★ 挂载点：直接指向宿主机 ~/tengee_workplace/verl
     └── data/     ← OverlayFS
```

`/workspace/verl` 不是 OverlayFS 的一部分 —— 它就是宿主机文件系统的目录，被 Linux 内核的 **bind mount** 机制嫁接到容器里。

### 完全双向同步

容器内写：

```bash
# 容器内
echo "hello" > /workspace/verl/test.txt
```

```bash
# 宿主机立即可见
cat ~/tengee_workplace/verl/test.txt
# hello
```

宿主机写也一样：

```bash
# 宿主机
echo "from host" > ~/tengee_workplace/verl/test.txt

# 容器内立即可见
cat /workspace/verl/test.txt
# from host
```

**它们就是同一个文件**，inode 完全一致：

```bash
# 容器内
stat /workspace/verl/README.md | grep Inode
# Inode: 123456

# 宿主机
stat ~/tengee_workplace/verl/README.md | grep Inode
# Inode: 123456  ← 相同
```

没有「拷贝」「同步」概念 —— 是同一块磁盘上的同一份字节。

### 性能优势

Volume 比容器内可写层快**很多**，尤其是：
- 数据集随机读（训练 IO 密集）
- 大文件写入（保存 checkpoint）
- 频繁的小文件操作

**所以训练数据集、模型、checkpoint 一定要放 volume**。

### 多容器共享

```bash
docker create --name verl_1 -v ~/data:/root/data ...
docker create --name verl_2 -v ~/data:/root/data ...
```

两个容器看到的 `/root/data` 是同一份。

---

## pip install -e 的本质

### 普通 install vs editable install

**普通 `pip install .`**：
1. 读 `setup.py` / `pyproject.toml`
2. 装依赖
3. **复制**整个 `verl/` 目录到 `site-packages/verl/`
4. `import verl` → 找到 site-packages 的副本

**editable `pip install -e .`**：
1. 读 `setup.py` / `pyproject.toml`
2. 装依赖
3. **不复制源码**，在 site-packages 写一个**路径指针文件**
4. `import verl` → 读指针 → 直接到原源码目录加载

**关键**：editable 模式**没有复制副本**，只有路径声明。

### 看看 editable 实际写了什么

```bash
ls /usr/local/lib/python3.12/site-packages/ | grep -i verl
# 输出（PEP 660 风格）：
# verl-0.7.x.dist-info/
# __editable__.verl-0.7.x.pth   ← 关键
# 或老 setuptools 风格：
# verl.egg-link
# easy-install.pth
```

cat 那个 `.pth` 文件：

```bash
cat /usr/local/lib/python3.12/site-packages/__editable__.verl-*.pth
# 内容就一行：
# /workspace/verl
```

整个 site-packages 里**没有任何 .py 源码副本**，只有这一条路径记录 + 几个 metadata 文件。

### Python 启动时怎么用这条记录

1. Python 解释器启动时自动扫描 `site-packages/` 下所有 `.pth` 文件
2. 把里面的路径加进 `sys.path`
3. `import verl` 按 `sys.path` 顺序查找 → 找到 `/workspace/verl/verl/__init__.py` → 加载

验证：

```bash
python3 -c "import sys; [print(p) for p in sys.path if 'workspace' in p]"
# /workspace/verl

python3 -c "import verl; print(verl.__file__)"
# /workspace/verl/verl/__init__.py   ← 直接指向源码
```

### 类比

|  | 普通 install | editable install |
|---|---|---|
| 类比 | 把书的复印件放进图书馆 | 在图书馆登记薄上写「这本书在 X 教授办公室」 |
| 改原书 | 复印件不变 | 立即生效，因为图书馆里没副本 |
| 占空间 | 占两份 | 几 KB（只有路径记录） |
| 适合 | 部署、生产 | 开发、调试 |

### `--no-deps` 干什么

```bash
pip install --no-deps -e .
```

- 不带 `--no-deps`：解析依赖 + 装依赖 + 注册路径
- 带 `--no-deps`：**只注册路径**

为什么用：镜像里 torch / vllm / flash-attn 已经精心调过版本，让 pip 重新解析依赖会破坏组合。

---

## 完整链路：宿主机改代码 → 容器立即生效

```
宿主机 ~/tengee_workplace/verl/verl/trainer/main_ppo.py
   ↓ (Linux bind mount，inode 完全一致)
容器 /workspace/verl/verl/trainer/main_ppo.py
   ↑
   │ (Python import 时去这里找)
   │
site-packages/__editable__.verl-x.x.x.pth ──→ "/workspace/verl"
   ↑
sys.path 启动时读 .pth → 加进搜索路径
   ↑
import verl.trainer.main_ppo
```

整个过程**没有任何复制**，只有：
1. 容器创建时的一次 Linux bind mount
2. `pip install -e` 时写的一个 `.pth` 文件

### 工作流

```
1. VSCode 在宿主机改代码
   ~/tengee_workplace/verl/verl/trainer/ppo_trainer.py 修改
       ↓
2. bind mount 让容器立即看到
   /workspace/verl/verl/trainer/ppo_trainer.py 同步变化
       ↓
3. 容器内跑 python -m verl.trainer.main_ppo
   Python 读 .pth → 知道 verl 包在 /workspace/verl
   import verl.trainer.ppo_trainer → 加载新代码
```

不需要重启容器，不需要重新 `pip install`。

### 验证「editable 实时生效」

```bash
# 容器内
echo "print('!!! editable works !!!')" >> /workspace/verl/verl/__init__.py
python3 -c "import verl"
# 输出：!!! editable works !!!

# 撤回（因为是挂载，宿主机也变了）
sed -i '$d' /workspace/verl/verl/__init__.py
```

---

## 实用对照表

### 三种存储位置对比

|  | Image 只读层 | 容器可写层 | Bind mount / Volume |
|---|---|---|---|
| 位置 | `/var/lib/docker/overlay2/...` | `/var/lib/docker/containers/...` | 宿主机任意路径 |
| 是否双向 | N/A（只读） | 单向（容器→外用 `docker cp`） | **完全双向实时** |
| 删容器后 | 不受影响 | 丢失 | 不受影响 |
| 跨容器共享 | 是（同 image） | 否 | 是（多容器挂同一目录） |
| 性能 | CoW（写慢） | CoW（写慢） | **原生文件系统** |
| 适合放什么 | 基础环境、依赖 | 临时修改、缓存 | 源码、数据集、模型、日志 |

### 容器生命周期对可写层的影响

| 操作 | 容器内 pip 装的东西 | 挂载卷里的东西 |
|---|---|---|
| `exit` 退出 shell | ✅ 还在（容器没停） | ✅ 还在 |
| `docker stop verl` | ✅ 还在（只是停了） | ✅ 还在 |
| `docker start verl` 再进 | ✅ 还在 | ✅ 还在 |
| `docker rm verl` 删容器 | ❌ 丢了 | ✅ 还在 |
| 重新 `docker create` | ❌ 全新可写层 | ✅ 还在 |

**结论**：只要不 `docker rm`，容器里所有改动都在。

### editable 模式下「丢失」的影响

`pip install --no-deps -e .` 后：
- **代码本体**：在 `/workspace/verl`（挂载卷）→ 永不丢失
- **`.pth` metadata**：在容器可写层 → 删容器会丢

但**重新跑 `pip install --no-deps -e .` 是秒级**（只是写几个小文件），所以即使删容器也不可怕。

---

## 实用启动脚本

把启动逻辑保存成脚本，重建容器只要一键：

```bash
#!/bin/bash
# ~/start_verl.sh

docker start verl 2>/dev/null || {
    docker create --gpus all \
        --net=host --shm-size="32g" \
        --cap-add=SYS_ADMIN \
        -e HF_ENDPOINT=https://hf-mirror.com \
        -e HF_HUB_ENABLE_HF_TRANSFER=1 \
        -v ~/tengee_workplace/verl:/workspace/verl \
        -v ~/data:/root/data \
        -v ~/models:/root/models \
        -v ~/.cache/huggingface:/root/.cache/huggingface \
        --name verl \
        verlai/verl:vllm017.latest \
        sleep infinity
    docker start verl
    # 自动重装 verl editable
    docker exec verl bash -c "cd /workspace/verl && pip3 install --no-deps -e ."
}

docker exec -it verl bash
```

---

## 常见坑

| 症状 | 原因 | 解决 |
|---|---|---|
| `pip install .`（不加 -e）后改代码不生效 | 没用 editable 模式 | 用 `pip install -e .` |
| `pip install -e .` 后 torch/vllm 版本变了 | 没加 `--no-deps`，pip 升级了依赖 | 重装镜像版本 + 用 `--no-deps` |
| 宿主机看不到容器创建的文件属主 | 容器默认 root，UID 0 | `chown` 或用 `--user $(id -u):$(id -g)` |
| 挂载目录不存在导致权限问题 | Docker 自动创建为 root 所有 | 提前 `mkdir -p` |
| 容器内 `/workspace/verl` 是空的 | 没正确挂载 | 检查 `-v` 路径，确认源路径存在 |
| NCCL / shm 报错 | 默认 `--shm-size=64M` 太小 | 加 `--shm-size=32g` |

---

## 心智模型一句话总结

> **容器 = 只读镜像 + 可写层 + 若干「挂到宿主机的门」**
>
> - 只读镜像像光盘，改动走 CoW 到**可写层**，删容器就回到光盘初始态
> - 挂载的 volume 不是光盘也不是可写层，是**直通宿主机的后门**，双向实时同步，删容器不影响
> - `pip install -e` 不复制源码，只在 site-packages 写一条**路径声明**，让 Python 直接到原地读

设计容器时问自己两个问题：
1. **删了容器还需要吗？** → 需要 → 挂载 volume
2. **要跨容器共享吗？** → 要 → 挂载 volume；不要 → 随便放可写层

---

## 参考命令速查

```bash
# 看容器可写层大小
docker ps -s

# 看可写层修改了哪些文件
docker diff <container>

# 看挂载情况
docker inspect <container> | grep -A 5 Mounts

# 看镜像层
docker history <image>

# 看容器的环境变量（包括 -e 传入的）
docker exec <container> env

# 看容器内进程的真实 inode（验证 bind mount）
docker exec <container> stat /workspace/verl/README.md
stat ~/tengee_workplace/verl/README.md   # 宿主机对比

# 看 Python 实际从哪里加载 verl
docker exec <container> python3 -c "import verl; print(verl.__file__)"

# 看 .pth 文件内容（editable 路径声明）
docker exec <container> bash -c "cat /usr/local/lib/python3.12/site-packages/__editable__.verl-*.pth"
```
