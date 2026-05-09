import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
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

TEMPLATE = """You are presented with a problem, a section of an article that may contain the answer to the problem, and a previous memory. You should generate response in the following format:
- Output your thinking process in <thinking>your_thinking_process</thinking>.
- Read the provided section carefully and update the memory with the new information that helps to answer the problem in only one <update>the_updated_memory</update> action. Be sure to retain all relevant details from the previous memory while adding any new, useful information.
- If you notice partial key evidence that is not enough to answer the problem, also output only one `<recall>query</recall>` (e.g. `<recall>who's the president of the United States?</recall>`) to retrieve information in previous memories.

<problem> 
{prompt}
</problem>

<recalled_memory>
{recalled_memory}
</recalled_memory>

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

<recalled_memory>
{recalled_memory}
</recalled_memory>

<memory>
{memory}
</memory>

Your answer:
"""

NO_MEMORY = "No previous memory"
NO_RECALLED_MEMORY = "No memory was recalled."


def _parse_recall_query(text_response: str) -> "str | None":
    try:
        match = re.search(r'<recall>(.+?)</recall>', text_response)
        if match:
            return match.group(1)
    except (ValueError, TypeError):
        pass
    return None


def _parse_update_memory(text_response: str) -> "str | None":
    """Remove <recall> and <thinking>/<think> blocks, return cleaned text as memory."""
    try:
        cleaned = re.sub(r'<recall>.*?</recall>', '', text_response, flags=re.DOTALL)
        return cleaned.strip()
    except (ValueError, TypeError):
        return None


class TfidfRetriever:
    """
    A class to handle TF-IDF retrieval using a Hugging Face tokenizer.
    The vectorizer is fitted once upon initialization.
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.vectorizer = TfidfVectorizer(tokenizer=self._llm_tokenizer)

    def _llm_tokenizer(self, text):
        """
        Custom tokenizer method that uses the instance's tokenizer.
        """
        lower_text = text.lower()
        tokens = self.tokenizer.tokenize(lower_text)
        normalized_tokens = [token.replace('Ġ', '') for token in tokens]
        return normalized_tokens

    def retrieve(self, query, corpus, top_k=3):
        """
        Retrieves the top_k most similar documents for a given query.
        """
        if not query or not corpus:
            return [(None, 0.0) for _ in range(top_k)]
        if not isinstance(corpus, list):
            corpus = list(corpus)
        try:
            tfidf_matrix = self.vectorizer.fit_transform(corpus)
        except Exception as e:
            return [(None, 0.0) for _ in range(top_k)]
        
        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, tfidf_matrix).flatten()
        top_ids = np.argsort(sims)[::-1][:top_k]
        return [(corpus[i], sims[i]) for i in top_ids]
    
    def top1_retrieve(self, query, corpus):
        return self.retrieve(query, corpus, top_k=1)[0][0]


class ReMemR1_API(LLMBaseAPI):
    """
    ReMemR1 API agent.
    """

    def __init__(self, tokenizer_model: "str", model_name: "str", base_url: "str", api_key: "str", enable_thinking: "bool" = False,
                 chunk_size: "int" = 5000, max_context_len: "int" = 100000000000):
        super().__init__(model_name, base_url, api_key)
        self.enable_thinking = enable_thinking
        self.chunk_size = chunk_size
        self.max_context_len = max_context_len
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model, trust_remote_code=True)
        self.retriever = TfidfRetriever(self.tokenizer)

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
        recalled_memory = NO_RECALLED_MEMORY
        history_memory = set()
        trajectories = []

        # ── Recurrent chunked processing ──
        for i in range(0, len(input_ids), self.chunk_size):
            chunk = input_ids[i:i + self.chunk_size]
            chunk_text = self.tokenizer.decode(chunk)

            msg = TEMPLATE.format(
                prompt=question, chunk=chunk_text,
                memory=memory, recalled_memory=recalled_memory,
            )
            response = await self.query(
                messages=[{"role": "user", "content": msg}],
                enable_thinking=self.enable_thinking,
                discard_thinking=True,
                max_tokens = 2048
            )

            try:
                memory = _parse_update_memory(response)
                history_memory.add(memory)
                recall_query = _parse_recall_query(response)
                recalled_memory = NO_RECALLED_MEMORY if recall_query is None else self.retriever.top1_retrieve(recall_query, history_memory)

                trajectories.append({
                    "response": response,
                    "recall_query": recall_query,
                    "recalled_memory": recalled_memory,
                    "memory": memory,
                })
            except Exception:
                import traceback
                traceback.print_exc()
                return '', trajectories

        # ── Final answer ──
        msg_final = TEMPLATE_FINAL.format(
            prompt=question, memory=memory, recalled_memory=recalled_memory,
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
        except Exception:
            import traceback
            traceback.print_exc()

        return '', trajectories
