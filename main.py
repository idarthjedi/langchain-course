import dotenv
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage
from langgraph.graph import MessagesState, StateGraph, END

from nodes import run_agent_reasoning, tool_node

load_dotenv()

AGENT_REASON: str = "agent_reason"
ACT: str = "act"
LAST: int = -1

def should_continue(state: MessagesState) -> str:
    """
    Check if the last message is a tool call.
    :param state:
    :return:
    """
    if not state["messages"][LAST].tool_calls:
        return END
    return ACT

flow = StateGraph(MessagesState)

flow.add_node(AGENT_REASON, run_agent_reasoning)
flow.set_entry_point(AGENT_REASON)
flow.add_node(ACT, tool_node)

flow.add_conditional_edges(AGENT_REASON, should_continue, {
    END:END,
    ACT:ACT
})

flow.add_edge(ACT, AGENT_REASON)

app = flow.compile()
app.get_graph().draw_mermaid_png(output_file_path="flow.png")

if __name__ == "__main__":
    print("working test")
    res = app.invoke({"messages": [HumanMessage("What is the temperature in Richmond, Virginia. List it, and then triple it.")]})
    print(res["messages"][LAST].content)

