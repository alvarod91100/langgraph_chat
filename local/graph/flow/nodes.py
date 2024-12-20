from langchain_core.documents import Document
from embeddings.chroma import (
    vector_store
)
from ..llm.llm_chains import (
    retrieval_grader_chain, 
    answer_generator_chain,
    answer_denial_generator_chain,
    hallucination_grader_chain,
    answer_grader_chain,
    source_router_chain, 
    safety_censorer_chain,
)
from ..llm.tools import web_search_tool

retriever = vector_store.as_retriever()

# Nodes: they affect the state of the graph
def retrieve(state):
    """
    Retrieve documents from vectorstore

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    question = state["question"]

    # Retrieval
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}

def set_question_safe(state):
    """
    Grade question safety

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, safety, that contains LLM generation
    """
    question = state["question"]
    question_safety = "safe"
    return {"question": question, "question_safety": question_safety}

def set_question_unsafe(state):
    """
    Grade question safety

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, safety, that contains LLM generation
    """
    question = state["question"]
    question_safety = "unsafe"
    return {"question": question, "question_safety": question_safety}

def generate(state):
    """
    Generate answer using RAG on retrieved documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]

    # RAG generation
    generation = answer_generator_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}

def generate_denial(state):
    """
    Generate answer using RAG on retrieved documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATING DENIAL---")
    question = state["question"]
    question_safety = state["question_safety"]

    generation = answer_denial_generator_chain.invoke({"question": question, "question_safety": question_safety})
    return {"question": question, "generation": generation}


def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question
    If any document is not relevant, we will set a flag to run web search

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Filtered out irrelevant documents and updated web_search state
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    # Score each doc
    filtered_docs = []
    web_search = "No"
    for d in documents:
        score = retrieval_grader_chain.invoke(
            {"question": question, "document": d.page_content}
        )
        grade = score["score"]
        # Document relevant
        if grade.lower() == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        # Document not relevant
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            # We do not include the document in filtered_docs
            # We set a flag to indicate that we want to run web search
            web_search = "Yes"
            continue
    return {"documents": filtered_docs, "question": question, "web_search": web_search}


def web_search(state):
    """
    Web search based based on the question

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Appended web results to documents
    """

    print("---WEB SEARCH---")
    question = state["question"]
    documents = state.get("documents", [])

    # Web search
    docs = web_search_tool.invoke({"query": question})
    web_results = "CONTEXT FROM WEB SEARCH:" + "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)
    if documents is not None:
        documents.append(web_results)
    else:
        documents = [web_results]
    return {"documents": documents, "question": question}