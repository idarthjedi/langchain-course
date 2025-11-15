from langchain_classic import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

llm = ChatOllama(model="gpt-oss:120b")

prompt = hub.pull("rlm/rag-prompt")

generation_chain = prompt | llm | StrOutputParser()

