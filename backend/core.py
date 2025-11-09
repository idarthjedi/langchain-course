from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_classic import hub
from langchain_classic.chains.combine_documents import \
    create_stuff_documents_chain
from langchain_classic.chains.history_aware_retriever import \
    create_history_aware_retriever
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_ollama import ChatOllama
from langchain_openai import OpenAIEmbeddings

CHROMA_FILE: str = "data/document-helper"

load_dotenv()


def run_llm(query: str, chat_history: List[Dict[str, Any]] = []):

    if __name__ == "__main__":
        file_path = "../data/document-helper"
    else:
        file_path = "data/document-helper"

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    docsearch = Chroma(persist_directory=file_path, embedding_function=embeddings)
    chat = ChatOllama(verbose=True, temperature=0, model="gpt-oss:120b")

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    stuff_documents_chain = create_stuff_documents_chain(chat, retrieval_qa_chat_prompt)

    rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")
    history_aware_retriever = create_history_aware_retriever(
        llm=chat, retriever=docsearch.as_retriever(), prompt=rephrase_prompt
    )
    qa = create_retrieval_chain(
        retriever=history_aware_retriever,
        combine_docs_chain=stuff_documents_chain,
    )
    # as_retriever(search_type = "similarity", search_kwargs = {"k": 3})
    result = qa.invoke(input={"input": query, "chat_history": chat_history})

    new_result = {
        "query": result["input"],
        "result": result["answer"],
        "source_documents": result["context"],
    }
    return new_result


if __name__ == "__main__":
    res = run_llm(query="What is a LangChain Chain?")
    print(res["result"])
