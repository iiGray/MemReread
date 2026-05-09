import logging
from typing import Any
from uuid import uuid4

import torch
from omegaconf import DictConfig
from tensordict import TensorDict
from transformers import PreTrainedTokenizer, ProcessorMixin
from typing_extensions import override
from recurrent.async_utils import ChatCompletionProxy


from recurrent.interface import AsyncRAgent, RConfig, RRegister
from recurrent.utils import log_step, msg, clip_long_string
from recurrent.impls.async_memory import MemoryConfig, AsyncMemoryDataset

from recursive.interface import AsyncOutput, Memory, ConversationContainer, log_nnodes
from recursive.tools import *

from verl.protocol import DataProtoItem
from verl.trainer.ppo.ray_trainer import _timer
import verl.utils.torch_functional as verl_F


logger = logging.getLogger(__file__)
logger.setLevel('INFO')



class MemReread(AsyncRAgent):
    def __init__(self, proxy: ChatCompletionProxy, tokenizer: PreTrainedTokenizer, config: RConfig, rollout_config: DictConfig):
        super().__init__(proxy, tokenizer, config, rollout_config)
    @override
    async def rollout(self, gen_item: DataProtoItem) -> AsyncOutput:
        """
        tensor keys in output:
        - standard verl: "prompts", "responses", "input_ids", "attention_mask", "position_ids"
        - recurrent rl: "sample_index", "final_mask"
        > input_ids = torch.cat([prompts, responses], dim=1)
        """
        timing_raw = {}
        sample_index = gen_item.batch['sample_index'].item()
        context_length = gen_item.batch["context_length"].item()

        conversation_container = ConversationContainer()

        # context = self.tokenizer.decode(gen_item.batch["context_ids"], skip_special_tokens = True)
        context_chunks = [
            self.tokenizer.decode(gen_item.batch['context_ids'][i: i + self.config.chunk_size], skip_special_tokens=True)\
            for i in range(0, len(gen_item.batch['context_ids']), self.config.chunk_size)
        ]
        context = "".join(context_chunks)
        question = gen_item.non_tensor_batch["prompt"]

        kwargs = self.sampling_params(gen_item.meta_info)
        kwargs["max_completion_tokens"] = self.config.max_memorization_length

        # context_chunks = [k for k in context_chunks if k.strip()]

        recorded = await self.solve(sample_index, logger, question, context, context_chunks, timing_raw, conversation_container, **kwargs)

        flattened = conversation_container.flatten(
            self.rollout_config.max_convs
        )

        conversations = flattened.messages_list

        if sample_index == 0:
            log_step(logger, "Final", conversations[flattened.dtype_list.index("final_answer")])
            logger.info(f"Question: {gen_item.non_tensor_batch['question']}\nGolden: {gen_item.non_tensor_batch['golden_answer']}")
            logger.info(f"nodes: {recorded['nodes']}")


        sample_index = torch.full((len(conversations),), sample_index, dtype=torch.long)

        nnodes = torch.full((len(conversations),), recorded["nodes"], dtype = torch.long)
        # final_mask = torch.full((len(conversations),), False, dtype=torch.bool)
        # final_mask[-1] = True


        return AsyncOutput(
            conversations = conversations,
            sample_index = sample_index,
            nnodes = nnodes,
            repeats = flattened.repeats_mask,
            final_mask = flattened.final_mask,
            middle_mask = flattened.middle_mask,
            read_mask = flattened.read_mask,
            decompose_mask = flattened.decompose_mask,
            integration_mask = flattened.integration_mask,
            uids_array = flattened.uids_array,
            type_array = flattened.type_array,
            recorded = recorded,
            timing_raw = timing_raw
        )


    async def query(self, messages: "list[dict[str, str]]", enable_thinking: "bool", max_tokens,
                    timing_name: "str", timing_raw: "dict[str, float]", ** kwargs) -> str: 
        kwargs['chat_template_kwargs'] = dict(enable_thinking = enable_thinking)
        kwargs['max_completion_tokens'] = max_tokens
        with _timer(timing_name, timing_raw):
            completions, err = await self.proxy.get_chat_completions(
                messages=messages,
                ** kwargs
            )
        with _timer("mt_mics", timing_raw):
            if err: raise err
        
        choice = completions.choices[0]
        content = choice.message.content
        
        return content

    async def set_answer_via_memory(self, memory: "Memory", is_final: "bool" = False, **kwargs):
        messages = (get_final_answering_messages if is_final else get_answering_messages)(
            memory.info().question, 
            memory.info().value,
        )
        response = await self.query(messages, enable_thinking = False, max_tokens = memory.max_tokens['final_response'] if is_final else memory.max_tokens['middle_response'],
                                    timing_name = memory.timing_names['get_answer'], 
                                    timing_raw = memory.timing_raw,
                                    **kwargs)
        response = response.strip()
        messages += [dict(role = "assistant", content = response)]
        memory.add_to_container(messages = messages, dtype = 'final_answer' if is_final else 'middle_answer')
        memory.info().set_answer(response)

        if memory.sample_index == 0 and (not is_final):
            log_nnodes(f'Node: {memory.nid} - middle_answer', memory, messages)


    async def build_submemory_from(self, subnid: "int", memory: "Memory", question_stack: "list[str]", context: "str", context_chunks: "list[str]", depth: "int", beam_id: "int", org_question) -> "Memory":
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
            conversationContainer = memory.conversationContainer,
            max_tokens = memory.max_tokens,
            infos = dict(question_repeats = int(memory.info().current_subquestion() in (question_stack + [org_question] + memory.info().query_history()))),
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
        response = await self.query(messages, enable_thinking = self.rollout_config.update_thinking, 
                                    max_tokens = memory.max_tokens['memorize'],
                                    timing_name = memory.timing_names['update_from_submemory'], timing_raw = memory.timing_raw, **kwargs)
        response = response.split("</think>")[-1].strip()
        messages += [dict(role = 'assistant', content = response)]
        memory.add_to_container(messages, dtype = 'integration')

        memory.info().update(response, from_where = dict(quesion = memory.info().question, response = memory.info().response))
        memory.info().add_history(sub_memory.info())
        
        if memory.sample_index == 0:
            log_nnodes(f'Node: {memory.nid} - integration_{len(memory.info().qa_history) - 1}', memory, messages)

    async def dfs_progress(self, memory: "Memory", question_stack: "list[str]", context: "str", context_chunks: "list[str]", depth: "int", cnodes: "list[int]", org_question, **kwargs) -> "dict":
        ret = dict(
            question = memory.info().question,
            stack = question_stack,
            nodes = 1,
            sub = [],
            response = None
        )
        if depth < self.rollout_config.max_depth and cnodes[0] < self.rollout_config.max_nodes:
            # max_width = (self.rollout_config.max_width // 2 + 1) if question_stack else self.rollout_config.max_width
            max_width = self.rollout_config.max_width
            for beam_id in range(max_width):
                cnodes[0] += 1
                if (cnodes[0] > self.rollout_config.max_nodes) or (await memory.judge_answerable(question_stack + [memory.info().question, org_question])): break
                submemory: "Memory" = await self.build_submemory_from(cnodes[0] - 1, memory, question_stack + [memory.info().question], context, context_chunks, depth + 1, beam_id, org_question)
                sub_ret = await self.dfs_progress(submemory, question_stack + [memory.info().question], 
                                                  context, context_chunks, depth + 1, cnodes, org_question, **kwargs)
                
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
            timing_names = dict(
                update_from_submemory = 'mt_async_gen',
                get_answer = 'mt_async_gen',
                read_chunk = 'mt_async_gen',
                try_progress = 'mt_async_gen'
            ),
            timing_raw = timing_raw,
            conversationContainer = conversationContainer,
            max_tokens = dict(
                decompose = self.config.max_decomposition_length,
                memorize = self.config.max_memorization_length,
                middle_response = self.config.max_middle_response_length,
                final_response = self.config.max_final_response_length
            ),
            infos = dict(question_repeats = 0),
            ** kwargs
        ).init_from_context()
        
        recorded = await self.dfs_progress(memory, [], context, context_chunks, 0, [1], question, **kwargs)

        return recorded


REGISTER = RRegister(config_cls=MemoryConfig, dataset_cls=AsyncMemoryDataset, agent_cls=MemReread)
