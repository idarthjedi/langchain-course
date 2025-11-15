from dotenv import load_dotenv
from langgraph.graph import END, StateGraph
from graph.const import RETRIEVE, GRADE_DOCUMENTS, GENERATE, WEBSEARCH
from graph.nodes import generate, grade_documents, retrieve,web_search
from graph.state import GraphState
from graph.chains.answer_grader import answer_grader
from graph.chains.hallucination_grader import hallucination_grader
from graph.chains.router import question_router, RouteQuery



load_dotenv()


def grade_grounded_generation(state: GraphState) -> str:
    print("--- CHECKING FOR HALLUCINATIONS--- ")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )

    if hallucination_grade := score.binary_score:
        print("--- OUTPUT IS GROUNDED IN DOCUMENTS ---")
        print("--- GRADING OUTPUT AGAINST QUESTION ---")
        score = answer_grader.invoke({"question": question, "generation": generation})
        if answer_grade := score.binary_score:
            print("--- DECISION: OUTPUT ANSWERS QUESTION ---")
            return "useful"
        else:
            print("--- DECISION: OUTPUT DOESN'T ANSWER THE QUESTION ---")
            return "not useful"
    else:
        print("--- OUTPUT IS NOT GROUNDED IN DOCUMENTS, RESEARCHING ONLINE ---")
        return "not supported"

def decide_to_generate(state: GraphState):
    print("--- ASSESS GRADED DOCUMENTS ---")

    if state["web_search"]:
        print(
            "--- DECISION: NOT ALL DOCS ARE RELEVANT TO SEARCH QUERY ---"

        )
        return WEBSEARCH
    else:
        print("--- DECISION: DOCS RELEVANT, GENERATE ---")
        return GENERATE

def route_question(state: GraphState) -> str:
    print("--- ROUTE QUESTION ---")
    question = state["question"]
    source: RouteQuery = question_router.invoke({"question": question})
    if source.datasource == WEBSEARCH:
        print("--- ROUTE QUESTION TO WEBSEARCH ---")
        return WEBSEARCH
    elif source.datasource == "vectorstore":
        print("--- ROUTE QUESTION TO VECTORSTORE ---")
        return RETRIEVE


workflow = StateGraph(GraphState)

workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(GRADE_DOCUMENTS, grade_documents)
workflow.add_node(GENERATE, generate)
workflow.add_node(WEBSEARCH, web_search)

#workflow.set_entry_point(RETRIEVE)
workflow.set_conditional_entry_point(
    route_question,
    path_map={
        WEBSEARCH: WEBSEARCH,
        RETRIEVE: RETRIEVE,
    }
)
workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)
workflow.add_conditional_edges(
    GRADE_DOCUMENTS,
    decide_to_generate,
    path_map={
        WEBSEARCH: WEBSEARCH,
        GENERATE: GENERATE,
    },
)
workflow.add_conditional_edges(
    GENERATE,
    grade_grounded_generation,
    path_map={
        "not supported": GENERATE,
        "useful": END,
        "not useful": WEBSEARCH,
    }
)
workflow.add_edge(WEBSEARCH, GENERATE)
workflow.add_edge(GENERATE, END)

app = workflow.compile()
app.get_graph().draw_mermaid_png(output_file_path="flow-1.png")
