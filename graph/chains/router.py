from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from langchain_ollama import ChatOllama
from tornado.routing import Router


class RouteQuery(BaseModel):
    """ Route a user query to the most relevant datasource """

    datasource: Literal["vectorstore", "websearch"] = Field(
        ...,
        description="Given a user question, chose to route it to a web search or vectorstore",
    )

llm = ChatOllama(model="gpt-oss:120b")
structured_llm_router = llm.with_structured_output(RouteQuery)

system = """You are an expert at routing a user question to a vectorstore or web search. \n
The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks.
Use the vectorstore for questions on those topics, for everything else, use the websearch."""

route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)

question_router = route_prompt | structured_llm_router


