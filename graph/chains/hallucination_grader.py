from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableSequence
from langchain_ollama import ChatOllama

from graph.chains.generations import generation_chain
from ingestion import retriever

llm = ChatOllama(model="gpt-oss:120b")

class GradeHallucination(BaseModel):
    """Binary scorer for hallucination present in generation answer"""

    binary_score: bool = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )

structured_llm_grader = llm.with_structured_output(GradeHallucination)

system_prompt = """You are a grader assessing whether an LLM generation is grounded in or supported by a set of documents. \n
Give a binary score of 'yes' or 'no', where 'yes' means that the answer is grounded in or supported by the facts."""

hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "Set of facts:\n\n{documents}\n\nLLM generation: {generation}"),
    ]
)

hallucination_grader: RunnableSequence = hallucination_prompt | structured_llm_grader

if __name__ == "__main__":
    question = "agent memory"
    docs = retriever.invoke(question)
    generation = generation_chain.invoke({"context": docs, "question": question})
    res: GradeHallucination  = hallucination_grader.invoke(
        {"documents": docs, "generation": generation}
    )

    pass
