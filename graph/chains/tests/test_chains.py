from dotenv import load_dotenv
from pprint import pprint

from graph.chains.router import question_router

load_dotenv()

from graph.chains.retreival_grader import GradeDocuments, retrieval_chain
from graph.chains.generations import generation_chain
from ingestion import retriever
from graph.chains.hallucination_grader import hallucination_grader, GradeHallucination
from graph.chains.router import question_router, RouteQuery

def test_retrieval_chain_answer_yes() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    doc_txt = docs[0].page_content

    res: GradeDocuments = retrieval_chain.invoke(
        {"question": question, "document": doc_txt}
    )

    assert res.binary_score == 'yes'

def test_retrieval_chain_answer_no() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    doc_txt = docs[0].page_content

    res: GradeDocuments = retrieval_chain.invoke(
        {"question": "Is New York Style pizza better than Chicago", "document": doc_txt}

    )

    assert res.binary_score == 'no'

def test_generation_chain() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    generation = generation_chain.invoke({"context": docs, "question": question})
    pprint(generation)

def test_hallucination_grader_answer_yes() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    generation = generation_chain.invoke({"context": docs, "question": question})
    res: GradeHallucination  = hallucination_grader.invoke(
        {"documents": docs, "generation": generation}
    )
    assert res.binary_score

def test_hallucination_grader_answer_no() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    #generation = generation_chain.invoke({"context": docs, "question": question})
    res: GradeHallucination  = hallucination_grader.invoke(
        {"documents": docs, "generation": "The best pizza depends on where you grew up"}
    )
    assert not res.binary_score

def test_router_to_vectorstore() -> None:
    question = "agent memory"

    res: RouteQuery = question_router.invoke({"question": question})
    assert res.datasource == "vectorstore"

def test_router_to_websearch() -> None:
    question = "What is the average price of pizza in Virginia?"

    res: RouteQuery = question_router.invoke({"question": question})
    assert res.datasource == "websearch"