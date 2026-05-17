#!/bin/bash
# Hermeneutic Search: Inference evaluation (no RL)
# Launches sglang model server on GPUs 5,6,7, then runs eval
# Usage: bash run_eval_inference.sh

set -x

export CUDA_VISIBLE_DEVICES=5,6,7
export CUDA_DEVICE_ORDER=PCI_BUS_ID

MODEL_PATH=/workspace/models/Qwen2.5-3B
RETRIEVER_URL=http://localhost:8000/retrieve
MODEL_PORT=8080
NUM_SAMPLES=100
MAX_CYCLES=5

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Start sglang server (3 GPUs, TP=1 since 3B is small)
echo "Starting sglang server on GPUs 5,6,7..."
python3 -m sglang.launch_server \
    --model-path $MODEL_PATH \
    --port $MODEL_PORT \
    --tp 1 \
    --mem-fraction-static 0.8 \
    --trust-remote-code &
SERVER_PID=$!

# Wait for server to be ready
echo "Waiting for model server..."
for i in $(seq 1 60); do
    if curl -s http://localhost:${MODEL_PORT}/health | grep -q "ok"; then
        echo "Server ready!"
        break
    fi
    sleep 5
done

# Check retriever is up
if ! curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "WARNING: Retriever at localhost:8000 not responding. Make sure BM25 retriever is running."
fi

# Run evaluation
python3 $SCRIPT_DIR/inference/eval_hermeneutic.py \
    --model_url http://localhost:${MODEL_PORT}/v1 \
    --retriever_url $RETRIEVER_URL \
    --data_path /workspace/verl/data/searchR1_processed/test.parquet \
    --max_cycles $MAX_CYCLES \
    --num_samples $NUM_SAMPLES \
    --output $SCRIPT_DIR/eval_results_base_3b.json \
    --verbose

echo "Evaluation complete. Shutting down server..."
kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null
echo "Done."
