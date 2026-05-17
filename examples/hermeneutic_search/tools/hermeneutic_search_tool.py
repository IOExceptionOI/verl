"""
Hermeneutic Search Tool for verl multi-turn rollout.

Key difference from standard SearchTool:
- Supports two actions: "search" (retrieval) and "transform" (context reset)
- On transform: returns a signal that tells the rollout to reset context to the new question only
- This integrates with verl's BaseTool interface for use in RL training

For RL training, the tool_config.yaml should reference this class.
"""

import json
import logging
import os
import re
from typing import Any, Optional
from uuid import uuid4

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse
from verl.tools.utils.search_r1_like_utils import perform_single_search_batch

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class HermeneuticSearchTool(BaseTool):
    """
    A tool that handles both search and transform actions for Hermeneutic Search.
    
    In the hermeneutic paradigm:
    - search: standard retrieval, returns information
    - transform: signals context reset; the new question becomes the sole context
    
    The transform action is handled at the rollout level (context reset),
    but this tool provides the search functionality and reward calculation.
    """

    def __init__(self, config: dict, tool_schema: Optional[OpenAIFunctionToolSchema] = None):
        if tool_schema is None:
            tool_schema = OpenAIFunctionToolSchema.model_validate({
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search for information relevant to the current question.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query_list": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "A list of search queries.",
                            }
                        },
                        "required": ["query_list"],
                    },
                },
            })
        super().__init__(config, tool_schema)
        
        self.retrieval_service_url = config.get("retrieval_service_url", "http://127.0.0.1:8000/retrieve")
        self.topk = config.get("topk", 3)
        self.timeout = config.get("timeout", 30)
        
        # Per-instance state: track cycles and transforms for reward
        self._instances: dict[str, dict] = {}

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        
        self._instances[instance_id] = {
            "question": kwargs.get("question", ""),
            "ground_truth": kwargs.get("ground_truth", []),
            "num_searches": 0,
            "num_transforms": 0,
            "transform_questions": [],
        }
        return instance_id, ToolResponse()

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        state = self._instances.get(instance_id, {})
        query_list = parameters.get("query_list", [])
        
        if not query_list:
            return ToolResponse(text="No queries provided."), 0.0, {}
        
        result_text, metadata = perform_single_search_batch(
            retrieval_service_url=self.retrieval_service_url,
            query_list=query_list,
            topk=self.topk,
            timeout=self.timeout,
        )
        
        state["num_searches"] += 1
        
        # Parse and format
        try:
            result_json = json.loads(result_text)
            formatted = result_json.get("result", "No results found.")
        except json.JSONDecodeError:
            formatted = "Search error."
        
        return ToolResponse(text=formatted), 0.0, {"search_count": state["num_searches"]}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        """Reward is computed externally via the reward function, not here."""
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        self._instances.pop(instance_id, None)
