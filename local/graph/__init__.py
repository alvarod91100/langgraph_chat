from pprint import pprint
from typing import List
import time
from langchain_core.documents import Document
from typing_extensions import TypedDict
from IPython.display import Image, display
from langgraph.graph import START, END, StateGraph
from .flow.nodes import (
    retrieve, 
    generate,
    generate_denial,
    grade_documents,
    web_search, 
    set_question_safe,
    set_question_unsafe,
)
from .flow.edges import (
    route_question,
    decide_to_generate,
    grade_generation_v_documents_and_question, 
    safety_check_question
)
from datetime import datetime
now = datetime.now().strftime("%Y%m%d_%H%M%S")

# State
class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        question_safety: safety of answering question
        generation: LLM generation
        web_search: whether to add search
        documents: list of documents
    """

    question: str
    question_safety: str
    generation: str
    web_search: str
    documents: List[str]

workflow = StateGraph(GraphState)

# Define the nodesl. This adds the nodes, order does not matter
# workflow.add_node( node name, node function ) 
workflow.add_node("set_question_safe", set_question_safe)  # set question as safe to answer
workflow.add_node("set_question_unsafe", set_question_unsafe)  # set question as unsafe to answer
workflow.add_node("websearch", web_search)  # web search
workflow.add_node("retrieve", retrieve)  # retrieve
workflow.add_node("grade_documents", grade_documents)  # grade documents
workflow.add_node("generate", generate)  # generate
workflow.add_node("generate_denial", generate_denial)  # generate denial

# Build graph connecting nodes with edges, conditional edges and entry point
workflow.set_conditional_entry_point(
    safety_check_question, # conditional function 
    {
        # conditional result : next node
        "safe": "set_question_safe",
        "unsafe": "set_question_unsafe",
    },
)

workflow.add_edge("set_question_unsafe", "generate_denial")
workflow.add_edge("generate_denial", END)
workflow.add_conditional_edges(
    "set_question_safe", #node name
    route_question, # conditional function
    {
        # conditional result : next node
        "websearch": "websearch",
        "vectorstore": "retrieve",
    },
)
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "websearch": "websearch",
        "generate": "generate",
    },
)
workflow.add_edge("websearch", "generate")
workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "not supported": "generate",
        "useful": END,
        "not useful": "websearch",
    },
)

app = workflow.compile()
image = app.get_graph(xray=1).draw_mermaid_png()
with open(f"testing/graphs/graph_{now}.png", "wb") as f:
    f.write(image)