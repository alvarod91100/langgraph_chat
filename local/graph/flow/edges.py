from pprint import pprint
from typing import List
import time
from langchain_core.documents import Document
from typing_extensions import TypedDict
from langgraph.graph import END, StateGraph
from embeddings.chroma import (
    vector_store
)
from ..llm.llm_chains import (
    retrieval_grader_chain, 
    answer_generator_chain,
    hallucination_grader_chain,
    answer_grader_chain,
    source_router_chain, 
    safety_censorer_chain
)

retriever = vector_store.as_retriever()

# Conditional Edge: they route the flow of the graph
def safety_check_question(state):
    """
    Deny question if it is unsafe

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """

    print("---CHECKING QUESTION SAFETY---")
    question = state["question"]
    score = safety_censorer_chain.invoke({"question": question})
    # the if statement adds some unchangeable structure to the edge 
    # We cannot just return the answer of the chain becasue of possible irregularities in answers.
    if score["score"] == "safe":
        print("---QUESTION IS SAFE---")
        return "safe"
    else:
        print("---QUESTION IS UNSAFE---")
        return "unsafe"

# Conditional Edges
def route_question(state):
    """
    Route question to web search or RAG.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """

    print("---ROUTING QUESTION---")
    question = state["question"]
    source = source_router_chain.invoke({"question": question})
    if source["datasource"] == "web_search":
        print("---ROUTED QUESTION TO WEB SEARCH---")
        return "websearch"
    elif source["datasource"] == "vectorstore":
        print("---ROUTED QUESTION TO RAG---")
        return "vectorstore"


def decide_to_generate(state):
    """
    Determines whether to generate an answer, or add web search

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    print("---ASSESSING GRADED DOCUMENTS---")
    state["question"]
    web_search = state["web_search"]
    state["documents"]

    if web_search == "Yes":
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print(
            "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, INCLUDE WEB SEARCH---"
        )
        return "websearch"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"


# Conditional edge
def grade_generation_v_documents_and_question(state):
    """
    Determines whether the generation is grounded in the document and answers question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """

    print("---CHECKING HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = hallucination_grader_chain.invoke(
        {"documents": documents, "generation": generation}
    )
    grade = score["score"]

    # Check hallucination
    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        # Check question-answering
        print("---GRADE GENERATION vs QUESTION---")
        score = answer_grader_chain.invoke({"question": question, "generation": generation})
        grade = score["score"]
        if grade == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        pprint("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"