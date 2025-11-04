import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_text_splitters import CharacterTextSplitter

load_dotenv()

if __name__ == "__main__":
    print("ingesting...")

    # Paper retrieved from https://doi.org/10.3390/app15084414
    loader = PyPDFLoader("passkeys.pdf")
    document = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

    # retrieve the content from the PDF
    # content = "".join([x.page_content for x in document])
    # chunks = text_splitter.split_text(content)

    texts = text_splitter.split_documents(document)

    # embeddings = OllamaEmbeddings(model="qwen3-embedding:8b")

    ollama_ef = embedding_functions.OllamaEmbeddingFunction(
        model_name="qwen3-embedding:8b"
    )
    client = chromadb.PersistentClient(path="./data/chroma_langchain_db")
    collection = client.get_or_create_collection(
        name="passkeys_paper", embedding_function=ollama_ef
    )
    # vector_store = Chroma(collection_name="passkeys_paper", embedding_function=embeddings, persist_directory="./data/chroma_langchain_db",)

    ids = [f"doc_{i}" for i in range(len(texts))]
    metadatas = [
        {"source": "your_document.pdf", "page": chunk.metadata.get("page", "N/A")}
        for chunk in texts
    ]
    documents_to_add = [chunk.page_content for chunk in texts]

    collection.add(documents=documents_to_add, metadatas=metadatas, ids=ids)
    print(f"Added {len(texts)} chunks to ChromaDB.")

    print("splitting...")
