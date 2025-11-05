import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_classic import hub
from langchain_classic.chains.combine_documents import \
    create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain

load_dotenv()


def ingestion():
    print(" Loading...")

    loader = PyPDFLoader("passkeys.pdf")
    documents = loader.load()
    print(" Splitting...")

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50, separator="\n")

    docs = text_splitter.split_documents(documents=documents)

    embeddings = OllamaEmbeddings(model="qwen3-embedding:8b")
    vectorstore = FAISS.from_documents(documents=docs, embedding=embeddings)
    print(" Saving Embeddings...")
    vectorstore.save_local("data/faiss_local_react")


def retrieval():
    query = "What are some of the main challenges hindering passkey adoption?"

    embeddings = OllamaEmbeddings(model="qwen3-embedding:8b")

    new_vectorstore = FAISS.load_local("data/faiss_local_react", embeddings=embeddings,
                                       allow_dangerous_deserialization=True)

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(ChatOllama(model="gpt-oss:120b"), retrieval_qa_chat_prompt)
    retrieval_chain = create_retrieval_chain(
        retriever=new_vectorstore.as_retriever(
            search_type="similarity",
        ),
        combine_docs_chain=combine_docs_chain,
    )

    result = retrieval_chain.invoke(input={"input": query})
    print(result)


if __name__ == "__main__":
    ingestion()

    retrieval()