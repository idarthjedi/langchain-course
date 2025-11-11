from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from chains import generate_chain, reflect_chain

load_dotenv()


class MessageGraph(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


REFLECT: str = "reflect"
GENERATE: str = "generate"


def generation_node(state: MessageGraph):
    return {"messages": [generate_chain.invoke({"messages": state["messages"]})]}


def reflection_node(state: MessageGraph):
    res = reflect_chain.invoke({"messages": state["messages"]})
    return {"messages": [HumanMessage(content=res.content)]}


builder = StateGraph(state_schema=MessageGraph)
builder.add_node(GENERATE, generation_node)
builder.add_node(REFLECT, reflection_node)
builder.set_entry_point(GENERATE)


def should_continue(state: MessageGraph):
    if len(state["messages"]) > 6:
        return END
    return REFLECT


builder.add_conditional_edges(
    GENERATE, should_continue, path_map={END: END, REFLECT: REFLECT}
)
builder.add_edge(REFLECT, GENERATE)

graph = builder.compile()
graph.get_graph().draw_mermaid_png(output_file_path="flow.png")

if __name__ == "__main__":
    print("start up...")

    inputs = HumanMessage(
        content="""
    Provide feedback on this paragraph:
    "Another identified risk in part one was pragmatic resistance. Although employees align around the importance of increasing security and protection, management measures their overall success by their ability to deliver products to market on time, not by their enthusiasm for adopting new authentication methods. It is highly likely, then, that employees will attempt to circumvent any change that impacts their ability to deliver products on time. Thus, to help increase change readiness and address program-related concerns, SimpleTech will divide the program into two major phases spanning a two-year timeline: an introductory phase (year one) and a mandatory phase (year two)."
    """
    )

    graph.invoke(inputs)
