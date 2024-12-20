from langchain_ollama import ChatOllama
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os, yaml
load_dotenv()

with open(os.getenv("PROMPTS_PATH"), 'r') as file:
    llm_prompts = yaml.safe_load(file).get("prompts")


# =====================================
# ========= RETRIEVAL GRADER ==========
# =====================================
retrieval_grader_specs = llm_prompts.get("retrieval_grader")
retrieval_grader_llm = ChatOllama(
    model=os.getenv("LLM_MODEL"), 
    format="json", 
    temperature=0
)
retrieval_grader_prompt = PromptTemplate(
    template = retrieval_grader_specs["prompt"],
    input_variables = retrieval_grader_specs["inputs"],
)
retrieval_grader_chain = retrieval_grader_prompt | retrieval_grader_llm | JsonOutputParser()


# =====================================
# ========= ANSWER GENERATOR ==========
# =====================================
answer_generator_specs = llm_prompts.get("answer_generator")
answer_generator_llm = ChatOllama(
    model=os.getenv("LLM_MODEL"), 
    temperature=0
)
answer_generator_prompt = PromptTemplate(
    template= answer_generator_specs["prompt"],
    input_variables= answer_generator_specs["inputs"]
)
answer_generator_chain = answer_generator_prompt | answer_generator_llm | StrOutputParser()

# =====================================
# ===== ANSWER DENIAL GENERATOR =======
# =====================================
answer_denial_generator_specs = llm_prompts.get("answer_denial_generator")
answer_denial_generator_prompt = PromptTemplate(
    template= answer_denial_generator_specs["prompt"],
    input_variables= answer_denial_generator_specs["inputs"]
)
answer_denial_generator_llm = ChatOllama(
    model=os.getenv("LLM_MODEL"), 
    temperature=0
)
answer_denial_generator_chain = answer_denial_generator_prompt | answer_denial_generator_llm | StrOutputParser()

# =====================================
# ======== HALLUCINATION GRADER =======
# =====================================
hallucination_grader_specs = llm_prompts.get("hallucination_grader")
hallucination_grader_llm = ChatOllama(
    model=os.getenv("LLM_MODEL"), 
    format="json",
    temperature=0
)
hallucination_grader_prompt = PromptTemplate(
    template= hallucination_grader_specs["prompt"],
    input_variables= hallucination_grader_specs["inputs"]
)
hallucination_grader_chain = hallucination_grader_prompt | hallucination_grader_llm | JsonOutputParser()


# =====================================
# ========== ANSWER GRADER ============
# =====================================
answer_grader_specs = llm_prompts.get("answer_grader")
answer_grader_llm = ChatOllama(
    model=os.getenv("LLM_MODEL"), 
    format="json",
    temperature=0
)
answer_grader_prompt = PromptTemplate(
    template= answer_grader_specs["prompt"],
    input_variables= answer_grader_specs["inputs"]  
)
answer_grader_chain = answer_grader_prompt | answer_grader_llm | JsonOutputParser()


# =====================================
# ========== SOURCE ROUTER ============
# =====================================
source_router_specs = llm_prompts.get("source_router")
source_router_llm = ChatOllama(
    model=os.getenv("LLM_MODEL"), 
    format="json",
    temperature=0
)
source_router_prompt = PromptTemplate(
    template = source_router_specs["prompt"],
    input_variables = source_router_specs["inputs"]
)
source_router_chain = source_router_prompt | source_router_llm | JsonOutputParser()

# =====================================
# ========= SAFETY CENSORER ===========
# =====================================
safety_censorer_specs = llm_prompts.get("safety_censorer")
safety_censorer_llm = ChatOllama(
    model=os.getenv("LLM_MODEL"), 
    format="json", 
    temperature=0
)
safety_censorer_prompt = PromptTemplate(
    template = safety_censorer_specs["prompt"],
    input_variables = safety_censorer_specs["inputs"],
)
safety_censorer_chain = safety_censorer_prompt | safety_censorer_llm | JsonOutputParser()