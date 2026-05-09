
from recursive.prompts import *
from recursive.utils import *


def get_reading_messages(question: "str", reference_memory: "str", memory: "str", chunk: "str"):
    return [dict(
        role = 'user',
        content = MEMAGENT_READING_TEMPLATE.format(
            prompt = question,
            memory = memory,
            chunk = chunk
        )
    )]

def get_progress_or_answer_messages(decomposed_stack: "list[str]", question: "str", memory: "str", history: "list[dict[str, str]]"):
    TEMPLATE = DECOMPOSING_OR_ANSWERING_TEMPLATE

    if not decomposed_stack:
        stack_str = "EMPTY"
    else:
        stack_str = "\n".join(decomposed_stack)
        
    if not history:
        history_str = "NO HISTORY"
    else:
        
        history_str = "\n".join([f"<query>{x['question']}</query> <result>{x['answer']}</result>" for x in history])

    return [dict(
        role = 'user', 
        content = TEMPLATE.format(
            decomposition_stack = stack_str,
            question=question, 
            memory=memory,
            decomposed_history = history_str
        ))]


def get_answering_messages(question: "str", memory: "str"):
    return [dict(
        role = 'user', 
        content = ANSWERING_TEMPLATE.format(
            question=question, 
            memory=memory,
        ))]

def get_final_answering_messages(question: "str", memory: "str"):
    return [dict(
        role = 'user', 
        content = ANSWERING_TEMPLATE.format(
            question=question, 
            memory=memory,
        ))]


def get_integrating_messages(question: "str", memory: "str", submemory: "str", subquestion: "str", answer: "str"):
    return [dict(
        role = 'user', 
        content = INTEGRATING_TEMPLATE.format(
            question=question, 
            memory=memory,
            submemory = submemory,
            subquestion = subquestion,
            answer = answer
        ))]