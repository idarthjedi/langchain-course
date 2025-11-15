from typing import List, TypedDict, Annotated
import operator

from langchain_core.outputs import generation


class GraphState(TypedDict):
    """
    Represents the state of a graph at a given point in time.

    This data structure is used for storing information about the current
    state of a graph, including visited nodes.
    It serves to help algorithms track the progress and maintain the graph's
    properties during computation.

    Attributes:
        question: question
        generation: LLM Generation
        web_search: whether to add search
        documents: List of documents
    """

    question: Annotated[str, lambda x, y: y]
    generation: str
    web_search: bool
    documents: str #Annotated[List[str], operator.add] #List[str]

