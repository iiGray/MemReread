import re
import numpy as np
from transformers import AutoTokenizer
from dataclasses import dataclass, field
from exp.tools.llminfer import async_query_api, close_redis_semaphore


class LLMBaseAPI:
    def __init__(self, model_name: "str", base_url: "str", api_key: "str"):
        self.model_name = model_name
        self.base_url = base_url
        self.api_key = api_key

    async def query(self, messages: "str", tools: "list | None" = None, enable_thinking: "bool" = False, discard_thinking: "bool" = False,
    return_text: "bool" = True, return_dict: "bool" = False, max_tokens = 32768) -> "str | dict":
        tool_choice = "auto" if tools else None
        return await async_query_api(
            messages = messages, model_name = self.model_name, base_url = self.base_url, api_key  = self.api_key, enable_thinking = False,
            max_tokens = max_tokens, return_text = return_text
        )

TEMPLATE = """You are presented with a problem, a section of an article that may contain the answer to the problem, and a previous memory. Please read the provided section carefully and update the memory with the new information that helps to answer the problem. Be sure to retain all relevant details from the previous memory while adding any new, useful information.

<problem> 
{prompt}
</problem>

<memory>
{memory}
</memory>

<section>
{chunk}
</section>

Updated memory:
"""

TEMPLATE_FINAL = """You are presented with a problem and a previous memory. Please answer the problem based on the previous memory and put the answer in \\boxed{{}}.

<problem> 
{prompt}
</problem>

<memory>
{memory}
</memory>

Your answer:
"""

NO_MEMORY = "No previous memory"

def clip_long_string(string, max_length=2000):
    """Clip long string to a maximum length."""
    if not len(string) > max_length:
        return string
    target_len = max_length - len("\n\n...(truncated)\n\n")
    return string[: target_len // 2] + "\n\n...(truncated)\n\n" + string[-target_len // 2 :]


class MemAgent_API(LLMBaseAPI):
    
    def __init__(self, tokenizer_model: "str", model_name: "str", base_url: "str", api_key: "str", enable_thinking: "bool" = False,
                 chunk_size: "int" = 5000, max_context_len: "int" = 100000000000):
        super().__init__(model_name, base_url, api_key)
        self.enable_thinking = enable_thinking
        self.chunk_size = chunk_size
        self.max_context_len = max_context_len
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model, trust_remote_code=True)

    async def __call__(self, question: "str", context: "str", context_chunks, **kwargs) -> "tuple[str, list[dict]]":
        """
        Args:
            question:  the question to answer.
            context:   full context string.
            context_chunks: (unused, kept for interface compatibility)

        Returns:
            (response_text, trajectories_list)
        """
        input_ids = self.tokenizer.encode(context)
        max_len = self.max_context_len
        if len(input_ids) > max_len:
            input_ids = input_ids[:max_len // 2] + input_ids[-max_len // 2:]

        memory = NO_MEMORY
        history_memory = set()
        trajectories = []

        # ── Recurrent chunked processing ──
        for i in range(0, len(input_ids), self.chunk_size):
            chunk = input_ids[i:i + self.chunk_size]
            chunk_text = self.tokenizer.decode(chunk)

            msg = TEMPLATE.format(
                prompt=question, chunk=chunk_text,
                memory=memory,
            )
            response = await self.query(
                messages=[{"role": "user", "content": msg}],
                enable_thinking=self.enable_thinking,
                discard_thinking=True,
                max_tokens = 2048
            )

            try:
                memory = response

                history_memory.add(memory)

                trajectories.append({
                    "response": response,
                    "memory": memory,
                })
            except KeyboardInterrupt:
                raise
            except Exception:
                import traceback
                traceback.print_exc()
                return '', trajectories

        # ── Final answer ──
        msg_final = TEMPLATE_FINAL.format(
            prompt=question, memory=memory,
        )
        try:
            response = await self.query(
                messages=[{"role": "user", "content": msg_final}],
                enable_thinking=self.enable_thinking,
                discard_thinking=False,
                max_tokens = 2048
            )
            trajectories.append({
                "step": "final",
                "response": response,
            })
            return response, trajectories
        except KeyboardInterrupt:
            raise
        except Exception:
            import traceback
            traceback.print_exc()

        return '', trajectories
