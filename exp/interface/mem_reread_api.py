from exp.tools import *
from exp.tools.llminfer import role_template, async_query_api, close_redis_semaphore
from dataclasses import dataclass, field
from typing import Optional, Dict
from transformers import AutoTokenizer

from recursive.prompts import *
from recursive.utils import *
from recursive.tools import *

import logging
from typing import Any, TYPE_CHECKING
if TYPE_CHECKING:
    from exp.interface import ConversationContainer

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


@dataclass
class MemoryInfo:
    depth: "int"
    beam_id: "int"
    question: "str"
    question_stack: "list[str]" = field(default_factory = list)
    reference_values: "list[str]" = NO_REFERENCE_MEMORY
    value: "str" = NO_MEMORY
    
    init_histories: "list[str]" = field(default_factory = list)
    update_histories: "list[str]" = field(default_factory = list)
    answerable: "bool" = False
    response: "str" = None

    subquestions: "list[str]" = field(default_factory = list)

    question_stack: "list[str]" = field(default_factory = list)
    qa_history: "list[str]" = field(default_factory = list)

    def update(self, value: "str", is_init: "bool" = False, from_where = None):
        value = str(value)
        self.value = value
        if is_init:
            self.init_histories += [value]
        else:
            self.update_histories += [{"value": value, "from_where": from_where}]

    def set_answerable(self, value: "bool" = True):
        self.answerable = value
        if value:
            self.subquestions += [None]

    def add_subquestion(self, question: "str"):
        if question:
            self.subquestions.append(question)
    
    def add_history(self, sub_info: "MemoryInfo"):
        self.qa_history += [{"question": sub_info.question, "answer": sub_info.response}]

    def current_subquestion(self):
        if len(self.subquestions):
            return self.subquestions[-1]
    
    def current_value(self):
        return self.value

    def get_history_values(self):
        return self.init_histories

    def set_question_stack(self, questions: "list[str]"):
        self.question_stack = questions

    def set_answer(self, answer: "str"):
        self.response = answer


class Memory:
    def __init__(self, 
                 nid: "int",
                 sample_index: "int",
                 logger: "logging.Logger",
                 agent: "MemReread_API",
                 depth: "int",
                 beam_id: "int",
                 question: "str", 
                 question_stack: "list[str]", 
                 reference_values: "list[str]", 
                 value: "str", 
                 context: "str",
                 context_chunks: "list[str] | None" = None,
                 timing_names: "dict[str, str]" = None,
                 timing_raw: "dict" = None,
                 conversationContainer: "ConversationContainer" = None,
                 **kwargs):
        self.nid = nid
        self.sample_index = sample_index
        self.logger = logger
        self.agent = agent
        self._info = MemoryInfo(depth, beam_id, question, question_stack, reference_values, value)

        self.question = question
        self.context = context
        self.context_chunks = context_chunks 

        self.timing_names = timing_names
        self.timing_raw = timing_raw 

        self.conversationContainer = conversationContainer

        self.kwargs = kwargs

    def info(self) -> "MemoryInfo":
        return self._info

    # def add_to_container(self, messages: "list[dict[str, str]]", dtype: "Literal['read', 'decompose', 'integration', 'middle_answer', 'final_answer']"):
    #     self.conversationContainer.add(
    #         self.info().depth, self.info().beam_id, messages, dtype
    #     )

    async def try_progress(self, ret) -> "str":
        messages = get_progress_or_answer_messages(
            self.info().question_stack, 
            self.info().question, 
            self.info().value, 
            self.info().qa_history
        )
        response = await self.agent.query(
            messages, 
            enable_thinking = False,
            max_tokens = 8192,
        )

        ret['decompose_his'] += [response]


        # tool_calls = parse_function_calls_from_text(response)[0]
        # progressed = [k['arguments'] for k in tool_calls if k['name'] == DECOMPOSE_FN_NAME]
        progressed = subquestions_from_text(response)
        print("HIS:", self.info().qa_history)
        print("STK:", self.info().question_stack)
        # print(f"PG: {progressed}, {tool_calls}")
        progressed = [(next(iter(k.values())) if isinstance(k, dict) else k) for k in progressed if bool(k)]
        if not progressed: return ""
        return progressed[0]

    async def judge_answerable(self, ret) -> "bool":
        
        progressed = await self.try_progress(ret)

        self.info().set_answerable(not bool(progressed))
        self.info().add_subquestion(progressed)

        return self.info().answerable

    async def read_chunk(self, chunk: "str", i: "int"):
        assert chunk.strip() != "", f"Empty chunk_{i}: [<{chunk}>]"
        messages = get_reading_messages(
                self.info().question, 
                "",
                # self.info().reference_values[i], 
                self.info().value, chunk
            )
        response = await self.agent.query(
            messages = messages,
            enable_thinking = False,
            max_tokens = 1024,
        )
        response = response.split("</think>")[-1]

        messages += [dict(role = 'assistant', content = response)]


        self.info().update(response, is_init = True, from_where = f"chunk_{i}")

    # async def judge_early_stop(self) -> "bool":
    #     return False
    async def init_from_context(self, memory: "str" = None, qa_history = []) -> "Memory":
        '''
        init memory and judge it answerable/ or progress
        '''

        if memory is not None:
            self.info().update(memory, is_init = True, from_where = f"last_read")
            for q, r in qa_history:
                self.info().add_history(
                    MemoryInfo(question = q, response = r)
                )
            return self

        for i, chunk in enumerate(self.context_chunks):
            await self.read_chunk(chunk, i)
            # if await self.judge_early_stop(): break
        return self




class MemReread_API(LLMBaseAPI):
    def __init__(self, 
                    tokenizer_model: "str",
                 model_name: "str", base_url: "str", api_key: "str", enable_thinking: "bool" = False,
                 max_chunk: "int" = 5000,
                 max_depth: "int" = 1,
                 max_width: "int" = 2,
                 max_nodes: "int" = 2):
        super().__init__(model_name, base_url, api_key)
        self.max_chunk = max_chunk
        self.max_depth = max_depth
        self.max_width = max_width
        self.max_nodes = max_nodes
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
        self.enable_thinking = enable_thinking


    async def __call__(self, question: "str", context: "str", context_chunks, fa_memory: "str" = None, qa_history = []) -> "tuple[list[str], dict]":
        """
        return: the answer, whether the problem has been successfully solved.
        """

        
        memory = await Memory(
            nid = 0,
            sample_index = None,
            logger = None,
            agent = self,
            depth = 0,
            beam_id = 0,
            question = question, 
            question_stack = [],
            reference_values = [NO_REFERENCE_MEMORY] * len(context_chunks),
            value = NO_MEMORY,
            context = context, 
            context_chunks = context_chunks,
            timing_names = dict(),
            timing_raw = None,
            conversationContainer = None,
        ).init_from_context(fa_memory, qa_history)
        
        recorded = await self.dfs_progress(memory, [], context, context_chunks, 0, [1])

        return memory.info().response, recorded

        


    async def set_answer_via_memory(self, memory: "Memory", is_final: "bool" = False, **kwargs):
        messages = (get_final_answering_messages if is_final else get_answering_messages)(
            memory.info().question, 
            memory.info().value,
        )
        response = await self.query(messages, enable_thinking = False, discard_thinking = True, max_tokens = 1024)
        messages += [dict(role = "assistant", content = response)]
        memory.info().set_answer(response)


    async def build_submemory_from(self, subnid: "int", memory: "Memory", question_stack: "list[str]", context: "str", context_chunks: "list[str]", depth: "int", beam_id: "int") -> "Memory":
        return await Memory(
            nid = subnid,
            sample_index = memory.sample_index,
            logger = memory.logger,
            agent = self, 
            depth = depth,
            beam_id = beam_id,
            question = memory.info().current_subquestion(), 
            question_stack = question_stack, 
            reference_values = memory.info().get_history_values(), 
            value = NO_MEMORY, 
            context = context, 
            context_chunks = context_chunks,
            timing_names = memory.timing_names,
            timing_raw = memory.timing_raw,
            conversationContainer = memory.conversationContainer
        ).init_from_context()

    async def update_from_submemory(self, sub_memory: "Memory", memory: "Memory", **kwargs):
        '''
        update memory value from submemory question / response
        '''
        messages = get_integrating_messages(
            memory.info().question,
            memory.info().value,
            sub_memory.info().value,
            sub_memory.info().question,
            sub_memory.info().response
        )

        response = await self.query(messages, enable_thinking = False, discard_thinking = True, max_tokens = 1024)

        response = response.split("</think>")[-1]

        memory.info().update(response, from_where = dict(quesion = memory.info().question, response = memory.info().response))
        memory.info().add_history(sub_memory.info())


    async def dfs_progress(self, memory: "Memory", question_stack: "list[str]", context: "str", context_chunks: "list[str]", depth: "int", cnodes: "list[int]", **kwargs) -> "dict":
        ret = dict(
            question = memory.info().question,
            org_memory = memory.info().value,
            decompose_his = [],
            stack = question_stack,
            nodes = 1,
            sub = [],
            response = None
        )
        if depth < self.max_depth and cnodes[0] < self.max_nodes:
            for beam_id in range(self.max_width):
                cnodes[0] += 1
                if (cnodes[0] > self.max_nodes) or (await memory.judge_answerable(ret)): break
                submemory: "Memory" = await self.build_submemory_from(cnodes[0] - 1, memory, question_stack + [memory.info().question], context, context_chunks, depth + 1, beam_id)
                sub_ret = await self.dfs_progress(submemory, question_stack + [memory.info().question], 
                                                  context, context_chunks, depth + 1, cnodes, **kwargs)
                
                await self.update_from_submemory(submemory, memory, **kwargs)

                ret['sub'] += [
                    dict(
                        sub = sub_ret,
                        depth = depth + 1,
                        beam_id = beam_id + 1,
                        fa_memory = memory.info().value,
                    )
                ]
                ret['nodes'] += sub_ret['nodes']

        await self.set_answer_via_memory(memory, is_final = (depth == 0), **kwargs)
        ret["reference_memory"] = memory.info().reference_values
        ret["memory"] = memory.info().value
        ret["response"] = memory.info().response
        return ret


    async def solve(self, sample_index: "int", logger, question: "str", context: "str", context_chunks: "list[str]" = None, 
                    timing_raw: "dict" = None, conversationContainer: "ConversationContainer" = None, **kwargs) -> "dict[str, Any]":
        """
        return: the answer, whether the problem has been successfully solved.
        """
        
        memory = await Memory(
            nid = 0,
            sample_index = sample_index,
            logger = logger,
            agent = self,
            depth = 0,
            beam_id = 0,
            question = question,    
            question_stack = [],
            reference_values = [NO_REFERENCE_MEMORY] * len(context_chunks),
            value = NO_MEMORY,
            context = context, 
            context_chunks = context_chunks,
            timing_names = dict(),
            timing_raw = None,
            conversationContainer = None,
            ** kwargs
        ).init_from_context()
        
        recorded = await self.dfs_progress(memory, [], context, context_chunks, 0, [1], **kwargs)

        return memory.info().response, recorded