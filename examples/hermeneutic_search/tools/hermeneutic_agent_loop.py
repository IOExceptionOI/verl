"""
Hermeneutic Search Agent Loop with TRUE context reset.

Format: only <transform>...</transform> (lowercase, strict).
Parser is strict — model must learn the exact format via RL.

Design:
- Each cycle is generated with a fresh prompt (real context reset during rollout).
- All cycles are packed compactly into one response sequence (no PAD separator).
- Cycle boundary positions are recorded in extra_fields for downstream training.
- The custom AgentLoopWorker uses these boundaries to compute position_ids with
  per-cycle reset, which makes transformers' flash_attention detect packed sequences
  and apply block-diagonal attention isolation.

See varlen_cu_seqlens_explained.md for the full mechanism.
"""

import json
import logging
import os
import re
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopOutput,
    register,
)
from verl.experimental.agent_loop.tool_agent_loop import (
    AgentData,
    ToolAgentLoop,
)
from verl.utils.rollout_trace import rollout_trace_op
from verl.workers.rollout.replica import TokenOutput

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def parse_hermeneutic_action(text: str) -> tuple:
    """Parse model output — strict format only."""
    # 1. <tool_call>...</tool_call> for search
    match = re.search(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL)
    if match:
        return "tool_call", match.group(1).strip()

    # 2. <transform>...</transform> (strict lowercase only)
    match = re.search(r"<transform>(.*?)</transform>", text, re.DOTALL)
    if match:
        return "transform", match.group(1).strip()

    # 3. <answer>...</answer>
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if match:
        return "answer", match.group(1).strip()

    return "none", ""


@register("hermeneutic_agent")
class HermeneuticAgentLoop(ToolAgentLoop):
    """
    Hermeneutic Search agent loop with true context reset.

    Actions (strict format):
    - <tool_call>{"name":"search","arguments":{"query_list":[...]}}</tool_call> → search
    - <transform>refined question</transform> → context reset to new question
    - <answer>final answer</answer> → terminate

    Output:
    - response_ids: compact concatenation of all cycles (no separators)
    - extra_fields["cycle_boundary_positions"]: list of int, position in
      response_ids where each cycle starts (always starts with 0). Used by
      HermeneuticAgentLoopWorker to compute position_ids with per-cycle reset.
    """

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        print("[HermeneuticAgentLoop] run() invoked")
        messages = list(kwargs["raw_prompt"])

        multi_modal_data = await self.process_vision_info(messages)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")

        metrics = {}
        request_id = uuid4().hex
        tools_kwargs = kwargs.get("tools_kwargs", {})

        # Per-cycle storage
        cycles_data = []
        current_cycle_prompt_ids = None
        current_cycle_response_ids = []
        current_cycle_response_mask = []
        current_cycle_logprobs = []

        # First cycle prompt
        initial_prompt_ids = await self.apply_chat_template(messages, tools=self.tool_schemas)
        current_cycle_prompt_ids = list(initial_prompt_ids)

        agent_data = AgentData(
            messages=messages,
            image_data=images,
            video_data=videos,
            metrics=metrics,
            request_id=request_id,
            tools_kwargs=tools_kwargs,
        )
        agent_data.prompt_ids = list(initial_prompt_ids)

        total_response_tokens = 0
        max_total_response = self.response_length
        num_cycles = 0
        max_cycles = self.max_assistant_turns
        transform_questions = []

        terminated = False

        while not terminated and num_cycles < max_cycles:
            num_cycles += 1
            cycle_done = False

            while not cycle_done:
                if total_response_tokens >= max_total_response:
                    terminated = True
                    break

                # Generate
                output: TokenOutput = await self.server_manager.generate(
                    request_id=request_id,
                    prompt_ids=agent_data.prompt_ids,
                    sampling_params=sampling_params,
                    image_data=agent_data.image_data,
                    video_data=agent_data.video_data,
                )

                if not agent_data.extra_fields:
                    agent_data.extra_fields.update(output.extra_fields)

                response_ids = list(output.token_ids)
                response_logprobs = list(output.log_probs) if output.log_probs else [0.0] * len(response_ids)

                current_cycle_response_ids.extend(response_ids)
                current_cycle_response_mask.extend([1] * len(response_ids))
                current_cycle_logprobs.extend(response_logprobs)
                total_response_tokens += len(response_ids)

                agent_data.prompt_ids = agent_data.prompt_ids + response_ids

                # Parse action
                response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
                action, content = parse_hermeneutic_action(response_text)

                if action == "answer":
                    terminated = True
                    cycle_done = True

                elif action == "tool_call":
                    # Execute search
                    try:
                        parsed = json.loads(content)
                        queries = parsed.get("arguments", {}).get("query_list", [content])
                    except (json.JSONDecodeError, AttributeError):
                        queries = [content]

                    tool = self.tools.get("search")
                    if tool:
                        try:
                            kwargs_for_tool = tools_kwargs.get("search", {})
                            instance_id, _ = await tool.create(
                                create_kwargs=kwargs_for_tool.get("create_kwargs", {})
                            )
                            tool_response, _, _ = await tool.execute(
                                instance_id, {"query_list": queries}, agent_data=agent_data
                            )
                            await tool.release(instance_id)
                            obs_text = tool_response.text or "No results."
                        except Exception as e:
                            obs_text = f"Search error: {e}"
                    else:
                        obs_text = "Search tool not available."

                    obs_formatted = f"\n<tool_response>\n{obs_text}\n</tool_response>\n"
                    obs_ids = self.tokenizer.encode(obs_formatted, add_special_tokens=False)

                    current_cycle_response_ids.extend(obs_ids)
                    current_cycle_response_mask.extend([0] * len(obs_ids))
                    current_cycle_logprobs.extend([0.0] * len(obs_ids))
                    total_response_tokens += len(obs_ids)

                    agent_data.prompt_ids = agent_data.prompt_ids + obs_ids

                elif action == "transform":
                    # === TRUE CONTEXT RESET ===
                    print(f"[HermeneuticAgentLoop] TRANSFORM: {content[:80]}...")
                    cycles_data.append({
                        "prompt_ids": current_cycle_prompt_ids,
                        "response_ids": current_cycle_response_ids,
                        "response_mask": current_cycle_response_mask,
                        "logprobs": current_cycle_logprobs,
                    })
                    transform_questions.append(content)

                    # Fresh prompt from new question
                    new_messages = [
                        {"role": "system", "content": messages[0]["content"]},
                        {"role": "user", "content": content},
                    ]
                    new_prompt_ids = await self.apply_chat_template(
                        new_messages, tools=self.tool_schemas
                    )

                    # Reset
                    current_cycle_prompt_ids = list(new_prompt_ids)
                    current_cycle_response_ids = []
                    current_cycle_response_mask = []
                    current_cycle_logprobs = []
                    agent_data.prompt_ids = list(new_prompt_ids)
                    agent_data.messages = new_messages
                    cycle_done = True

                else:
                    # Invalid action — append feedback and let model retry (like SR1)
                    feedback = (
                        "\nMy previous action is invalid. "
                        "If I want to search, I should use <tool_call> with a search query. "
                        "If I want to refine the question, I should put the new question between <transform> and </transform>. "
                        "If I want to give the final answer, I should put the answer between <answer> and </answer>. "
                        "Let me try again.\n"
                    )
                    feedback_ids = self.tokenizer.encode(feedback, add_special_tokens=False)

                    current_cycle_response_ids.extend(feedback_ids)
                    current_cycle_response_mask.extend([0] * len(feedback_ids))
                    current_cycle_logprobs.extend([0.0] * len(feedback_ids))
                    total_response_tokens += len(feedback_ids)

                    agent_data.prompt_ids = agent_data.prompt_ids + feedback_ids

        # Save final cycle
        if current_cycle_response_ids:
            cycles_data.append({
                "prompt_ids": current_cycle_prompt_ids,
                "response_ids": current_cycle_response_ids,
                "response_mask": current_cycle_response_mask,
                "logprobs": current_cycle_logprobs,
            })

        # === Pack cycles compactly (no PAD separators) ===
        # Cycle 0: prompt is the original prompt (returned separately)
        #          response = cycle_0_response
        # Cycle i (i>=1): both prompt and response go into final_response_ids,
        #                  with cycle_i_prompt marked response_mask=0 (non-trainable)
        # We record cycle_boundary_positions: start position of each cycle in response_ids.
        final_prompt_ids = cycles_data[0]["prompt_ids"] if cycles_data else list(initial_prompt_ids)
        final_response_ids = []
        final_response_mask = []
        final_logprobs = []
        cycle_boundary_positions = [0]  # cycle 0 always starts at position 0

        for i, cycle in enumerate(cycles_data):
            if i == 0:
                # Cycle 0: only response goes into response_ids
                final_response_ids.extend(cycle["response_ids"])
                final_response_mask.extend(cycle["response_mask"])
                final_logprobs.extend(cycle["logprobs"])
            else:
                # Record where this cycle starts (relative to response_ids)
                cycle_boundary_positions.append(len(final_response_ids))

                # Cycle i prompt (non-trainable, in response_ids region)
                final_response_ids.extend(cycle["prompt_ids"])
                final_response_mask.extend([0] * len(cycle["prompt_ids"]))
                final_logprobs.extend([0.0] * len(cycle["prompt_ids"]))

                # Cycle i response (trainable)
                final_response_ids.extend(cycle["response_ids"])
                final_response_mask.extend(cycle["response_mask"])
                final_logprobs.extend(cycle["logprobs"])

        # Trace logging (like SR1's random sampling)
        import random as _rnd
        if _rnd.randint(1, 8) == 1:
            _text = self.tokenizer.decode(final_response_ids[:max_total_response], skip_special_tokens=False)
            print(f"[SAMPLE] cycles={num_cycles} transforms={len(transform_questions)} boundaries={cycle_boundary_positions}")
            print(f"[SAMPLE] {_text[:800]}")
            if transform_questions:
                print(f"[SAMPLE] Qs: {transform_questions[:3]}")

        output = AgentLoopOutput(
            prompt_ids=final_prompt_ids,
            response_ids=final_response_ids[:max_total_response],
            response_mask=final_response_mask[:max_total_response],
            multi_modal_data={},
            response_logprobs=final_logprobs[:max_total_response] if any(l != 0 for l in final_logprobs) else None,
            num_turns=num_cycles * 2,
            metrics=metrics,
            routed_experts=None,
            extra_fields={
                **agent_data.extra_fields,
                "num_cycles": num_cycles,
                "transform_questions": transform_questions,
                # Explicit cycle boundaries for HermeneuticAgentLoopWorker
                "cycle_boundary_positions": cycle_boundary_positions,
                "prompt_length": len(final_prompt_ids),
            },
        )
        return output
