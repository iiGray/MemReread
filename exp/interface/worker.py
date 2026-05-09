from exp.tools import *
from exp.tools.llminfer import role_template, async_query_deployed, close_redis_semaphore
import nltk, json, re
import bisect
from typing import Optional, Dict
from transformers import AutoTokenizer

class LLMBase:
    def __init__(self, model_name: "str", port: "int"):
        self.model_name = model_name
        self.port = port

    async def query(self, messages: "str", tools: "list | None" = None, enable_thinking: "bool" = False, discard_thinking: "bool" = False,
    return_text: "bool" = True, return_dict: "bool" = False, max_tokens = 32768) -> "str | dict":
        tool_choice = "auto" if tools else None
        return await async_query_deployed(
            messages = messages, model_name = self.model_name, port = self.port, enable_thinking = enable_thinking, discard_thinking = discard_thinking,
            max_tokens = max_tokens,
            tools = tools, tool_choice = tool_choice, return_text = return_text, return_dict = return_dict
        )