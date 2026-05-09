
MEMAGENT_READING_TEMPLATE = """You are presented with a problem, a section of an article that may contain the answer to the problem, and a previous memory. Please read the provided section carefully and update the memory with the new information that helps to answer the problem. Be sure to retain all relevant details from the previous memory while adding any new, useful information.

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


DECOMPOSING_OR_ANSWERING_TEMPLATE = """You are a Question Analysis and Query Generation Agent.
You will receive a QUESTION, a MEMORY containing known facts and evidence, and a QUERY_HISTORY containing previously submitted exploratory queries with their results. Your task is to analyze whether the information in the MEMORY is sufficient to fully answer the QUESTION. If it is insufficient, you must generate a single targeted exploratory query to fill the most critical gap.

Critical Rules & Analysis Steps:
1. Focus on QUESTION: Evaluate only whether MEMORY can answer the current QUESTION. Do not attempt to answer or re-query any other questions.
2. Compare needs with MEMORY: Break the QUESTION down into specific information needs and compare them with what the MEMORY already contains. Identify if any crucial facts are still missing, uncertain, or incomplete.
3. Review QUERY_HISTORY: Avoid repeating any query already asked and recorded in QUERY_HISTORY.
4. You MUST NOT answer the QUESTION directly. Your output must strictly follow the decision logic below.

Decision Logic & Actions:
- IF SUFFICIENT: If the MEMORY already contains all necessary information to fully address the QUESTION, you must stop immediately. Do not generate any query and do not generate any further text.
- IF INSUFFICIENT: If the MEMORY lacks necessary information for the QUESTION, you must leverage the known information in the MEMORY to identify and submit the single highest-priority exploratory query that must be resolved first to push the QUESTION forward.

Guidelines for the Priority Query (Only if MEMORY is insufficient for the QUESTION):
1. Highest Priority: Among all possible information gaps for the QUESTION, select only the one query whose resolution is most critical and foundational for answering the QUESTION.
2. No Repetition: The query must NOT duplicate any query already present in the QUERY_HISTORY. Review the history carefully and ensure your proposed query is novel and represents genuine progress.
3. Independence: The query must be answerable in isolation without requiring answers to other queries.
4. Sufficiency: Resolving this query, combined with the existing MEMORY, must meaningfully progress toward fully resolving the QUESTION.
5. Self-Contained Expression: The query must be fully self-contained and free of any references to the original QUESTION, MEMORY, or external context. Never use pronouns, option labels, or context-dependent phrases like "the entity mentioned above", "option A", or "this event". Instead, explicitly state all relevant entities, values, and content.
6. Output Format: Submit exclusively this single highest-priority query by wrapping it in <query> tags. Output exactly one query; never submit multiple queries.
7. Confirmed Information Exclusion: Do NOT generate a query targeting any information enclosed within <confirmed>...</confirmed> tags in the MEMORY. These tags denote facts, evidence, or confirmed absences that have already been explicitly verified and resolved. Only generate queries for information whose existence, value, or status remains uncertain, ambiguous, or entirely unaddressed in the MEMORY. This prevents redundant exploration of settled information and directs attention to genuinely unresolved gaps.

QUESTION that you need to focus on:
<question>
{question}
</question>

<memory>
{memory}
</memory>

<query_history>
{decomposed_history}
</query_history>

"""




ANSWERING_TEMPLATE = """You are presented with a problem and a previous memory. Please answer the problem based on the previous memory and put the answer in \\boxed{{}}.

<problem> 
{question}
</problem>

<memory>
{memory}
</memory>

Your answer:
"""



INTEGRATING_TEMPLATE = """You are a memory integration assistant.
You will receive three inputs: a QUESTION, a current MEMORY snippet, and a reference subquestion-answer pair obtained from the latest question progression step.

Your task is to integrate the information from the reference into the MEMORY so that the updated MEMORY progresses towards answering the original QUESTION.

Follow these rules strictly:
1. Do not answer the original QUESTION directly; your sole output is the integrated MEMORY.
2. When information in the reference conflicts with existing MEMORY content, prioritize the reference information as it represents fresh research.
3. Eliminate redundant information; if a fact already exists in MEMORY, do not add it again.
4. Filter out any information irrelevant to the original QUESTION; retain only content that contributes to answering it.
5. Express all integrated information in fluent, coherent natural language. Do not copy the reference verbatim; instead, extract key facts and synthesize them into descriptive prose.
6. Cross-Source Evidence Tagging: Wrap a statement in <confirmed>...</confirmed> tags if the exact same factual claim (regarding existence, occurrence, or verified absence of evidence) appears in BOTH the current <memory> input AND the <subanswer> section of the reference. This cross-source consistency indicates higher reliability. Do NOT tag information that appears in only one of these two sources, or that is speculative, uncertain, or inferred.
7. Preserve Existing Confirmed Tags: If the current <memory> already contains information enclosed in <confirmed>...</confirmed> tags, and this information is not contradicted by the reference, retain these tagged segments verbatim in the updated MEMORY. Integrate them naturally into the surrounding prose without removing or altering the tags.

Your final output should be a concise, well-structured MEMORY that consolidates all verified, relevant information needed to resolve the original QUESTION.

<question>
{question}
</question>

<memory>
{memory}
</memory>

<reference>
<subquestion>
{subquestion}
</subquestion>
<subanswer>
{answer}
</subanswer>
</reference>

Updated memory:
"""



NO_MEMORY = "No previous memory"
NO_REFERENCE_MEMORY = "No reference memory"

