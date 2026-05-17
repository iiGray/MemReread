import numpy as np
import torch, random, logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Literal
from dataclasses import dataclass, field

from recurrent.utils import log_step, msg, clip_long_string
from recursive.tools import *



if TYPE_CHECKING:
    from recursive.impls.mem_reread import MemReread

def log_nnodes(dtype, memory: "Memory", conversation):
    logger = memory.logger
    logger.info("="*25 + f"{dtype}  ==== Depth: {memory.info().depth} Beam {memory.info().beam_id}" + "="*25)
    for i, msg in enumerate(conversation):
        logger.info(f"[{msg['role']}]:")
        logger.info(f"{clip_long_string(msg['content'])}")
        logger.info("-"*50)


class AsyncOutput(ABC):
    def __init__(self, 
                 conversations: "list[list[dict[str, str]]]", 
                 sample_index: int, 
                 nnodes: int,
                 repeats: int,
                 final_mask: bool,
                 middle_mask: bool,
                 read_mask: bool,
                 decompose_mask: bool,
                 integration_mask: bool,
                 uids_array: str,
                 type_array:str,
                 recorded: dict,
                 timing_raw: dict,
                 metrics: dict = None):
        self.conversations = conversations
        self.sample_index = sample_index
        self.repeats = repeats
        self.nnodes = nnodes
        self.final_mask = final_mask
        self.middle_mask = middle_mask
        self.read_mask = read_mask
        self.decompose_mask = decompose_mask
        self.integration_mask = integration_mask
        self.uids_array = uids_array
        self.type_array = type_array
        self.recorded = recorded
        self.timing_raw = timing_raw
        if metrics is None:
            metrics = {}
        self.metrics = metrics
        if "workflow/num_conv" not in metrics:
            metrics["workflow/num_conv"] = len(conversations)
    


@dataclass
class Conversation:
    uid: "int"
    depth: "int"
    beam_id: "int"
    value: "list[dict[str, str]]"
    infos: "dict"
    dtype: "Literal['read_{{}}', 'decompose', 'integration', 'middle_answer', 'final_answer']"

@dataclass
class FlattenedConversations:
    messages_list: "list[list[dict[str, str]]]"
    dtype_list: "list[str]"
    repeats_list: int
    final_index: "int"
    middle_indice: "int"
    read_indice: "list[int]"
    decompose_indice: "list[int]"
    integration_indice: "list[int]"
    pos_uids: "list[str]"

    # final_mask: "torch.Tensor"
    # middle_mask: "torch.Tensor"
    # read_mask: "torch.Tensor"
    # decompose_mask: "torch.Tensor"
    # integration_mask: "torch.Tensor"

    def __post_init__(self):
        self.final_mask = torch.full((len(self.messages_list),), False, dtype=torch.bool)
        self.final_mask[self.final_index] = True

        self.middle_mask = torch.full((len(self.messages_list),), False, dtype=torch.bool)
        self.middle_mask[self.middle_indice] = True

        self.read_mask = torch.full((len(self.messages_list),), False, dtype=torch.bool)
        self.read_mask[self.read_indice] = True

        self.decompose_mask = torch.full((len(self.messages_list),), False, dtype=torch.bool)
        self.decompose_mask[self.decompose_indice] = True

        self.integration_mask = torch.full((len(self.messages_list),), False, dtype=torch.bool)
        self.integration_mask[self.integration_indice] = True

        self.uids_array = np.array(self.pos_uids)
        self.type_array = np.array(self.dtype_list)

        self.repeats_mask = torch.tensor(self.repeats_list).bool()


class ConversationContainer:
    def __init__(self):
        self.conversations: "list[Conversation]" = []
    
    def add(self, depth: "int", beam_id: "int", value: "list[dict[str, str]]", infos: "dict",
            dtype: "Literal['read', 'decompose', 'integration', 'middle_answer', 'final_answer']"):
        self.conversations.append(
            Conversation(
                f"d{depth}_b{beam_id}_type{dtype}", depth, beam_id, value, infos, dtype
            )
        )

    def filtered_convs(self, dtype: "Literal['read', 'decompose', 'integration', 'middle_answer', 'final_answer']"):
        return [c for c in self.conversations if dtype in c.dtype]

    def flatten(self, max_convs: "int") -> "FlattenedConversations":
        '''
        return: 
            list of dict[str, str]: flattened conversations
            int: final pos
            list:int, 
        '''

        conversations = [k for k in self.conversations if (k.infos['question_repeats']== 0) or (k.dtype == 'decompose') ]

        if max_convs <= 0:
            must = conversations
        else:
            must_type = ('decompose', 'final_answer')
            must = [k for k in conversations if k.dtype in must_type]
            others = [k for k in conversations if k.dtype not in must_type]

            must += random.sample(others, min(max(0, max_convs - len(must)), len(others)))

        dtype_list = [k.dtype for k in must]
        messages_list  = [k.value for k in must]
        repeats_list = [k.infos['question_repeats'] for k in must]

        pos_uids = [k.uid for k in must]


        return FlattenedConversations(
            messages_list = messages_list,
            dtype_list = dtype_list,
            repeats_list = repeats_list,
            final_index = dtype_list.index('final_answer'),
            middle_indice = [i for i, k in enumerate(dtype_list) if 'middle_answer' in k],
            read_indice = [i for i, k in enumerate(dtype_list) if 'read' in k ],
            decompose_indice = [i for i, k in enumerate(dtype_list) if 'decompose' in k],
            integration_indice = [i for i, k in enumerate(dtype_list) if 'integration' in k],
            pos_uids = pos_uids
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
    
    def query_history(self):
        return [k['question'] for k in self.qa_history]

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
                 agent: "MemReread",
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
                 max_tokens: dict = None,
                 infos: dict = dict(),
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

        self.max_tokens = max_tokens

        self.infos = infos

        self.kwargs = kwargs

    def info(self) -> "MemoryInfo":
        return self._info

    def add_to_container(self, messages: "list[dict[str, str]]", dtype: "Literal['read', 'decompose', 'integration', 'middle_answer', 'final_answer']", infos = None):
        if infos is None: infos = self.infos
        self.conversationContainer.add(
            self.info().depth, self.info().beam_id, messages, infos, dtype
        )

    async def try_progress(self, question_his) -> "str":
        messages = get_progress_or_answer_messages(
            self.info().question_stack, 
            self.info().question, 
            self.info().value, 
            self.info().qa_history
        )
        response = await self.agent.query(
            messages, 
            enable_thinking = False, 
            max_tokens = self.max_tokens['decompose'],
            timing_name = self.timing_names['try_progress'],
            timing_raw = self.timing_raw,
            ** self.kwargs
        )

        response = response.strip()

        messages += [dict(role = 'assistant', content = response)]


        if self.sample_index == 0:
            log_nnodes(f'Node: {self.nid} - decompose_{len(self.info().qa_history)}', self, messages)


        progressed = subquestions_from_text(response)
        # print("HIS:", self.info().qa_history)
        # print("STK:", self.info().question_stack)
        progressed = [(next(iter(k.values())) if isinstance(k, dict) else k) for k in progressed if bool(k)]
        if not progressed: 
            self.add_to_container(messages, "decompose")
            return ""
        if progressed[0] in question_his:

            self.add_to_container(messages, "decompose", infos = dict(question_repeats = 1))

        return progressed[0]

    async def judge_answerable(self, question_his) -> "bool":
        
        progressed = await self.try_progress(question_his)

        self.info().set_answerable(not bool(progressed))
        self.info().add_subquestion(progressed)

        return self.info().answerable

    async def read_chunk(self, chunk: "str", i: "int"):
        # assert chunk.strip() != "", f"Empty chunk_{i}: [<{chunk}>]"
        messages = get_reading_messages(
                self.info().question, 
                self.info().reference_values[i], 
                self.info().value, chunk
            )
        response = await self.agent.query(
            messages = messages,
            enable_thinking = False,
            max_tokens = self.max_tokens['memorize'],
            timing_name = self.timing_names['read_chunk'],
            timing_raw = self.timing_raw,
            ** self.kwargs
        )
        response = response.split("</think>")[-1].strip()

        messages += [dict(role = 'assistant', content = response)]

        self.add_to_container(messages, f'read_{i}')

        if self.sample_index == 0:
            log_nnodes(f'Node: {self.nid} - read_{i}', self, messages)

        self.info().update(response, is_init = True, from_where = f"chunk_{i}")

    # async def judge_early_stop(self) -> "bool":
    #     return False
    async def init_from_context(self) -> "Memory":
        '''
        init memory and judge it answerable/ or progress
        '''
        for i, chunk in enumerate(self.context_chunks):
            await self.read_chunk(chunk, i)
            # if await self.judge_early_stop(): break
        return self

