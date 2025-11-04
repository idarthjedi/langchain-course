import os

import chromadb as db
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_classic import hub
from langchain_classic.chains.combine_documents import \
    create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings

load_dotenv()

if __name__ == "__main__":
    print(" Retrieving...")

    embeddings = OllamaEmbeddings(model="qwen3-embedding:8b")
    llm = ChatOllama(model="gpt-oss:120b")

    query = "What are some of the main challenges hindering passkey adoption?"
    # chain = PromptTemplate.from_template(template=query) | llm
    # result = chain.invoke(input={})
    # print(result.content)

    client = db.PersistentClient(path="./data/chroma_langchain_db")
    vector_store = Chroma(
        client=client, collection_name="passkeys_paper", embedding_function=embeddings
    )

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    retrieval_chain = create_retrieval_chain(
        retriever=vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 3}
        ),
        combine_docs_chain=combine_docs_chain,
    )

    result = retrieval_chain.invoke(input={"input": query})
    print(result)

    pass
