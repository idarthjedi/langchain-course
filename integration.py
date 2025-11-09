import asyncio
import os
import ssl
from typing import Any, Dict, List

import certifi
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap
from langchain_text_splitters import RecursiveCharacterTextSplitter

from logger import (Colors, log_error, log_header, log_info, log_success,
                    log_warning)

load_dotenv()

ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()


# embeddings = OllamaEmbeddings(model="snowflake-arctic-embed") #qwen3-embedding:8b")

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    show_progress_bar=True,
    chunk_size=50,
    retry_min_seconds=10,
)

chroma = Chroma(persist_directory="data/document-helper", embedding_function=embeddings)
tavily_extract = TavilyExtract()
tavily_map = TavilyMap(max_depth=5, max_breadth=20, max_pages=1000)
tavily_crawl = TavilyCrawl()


async def index_documents_async(documents: List[Document], batch_size: int = 50):
    """Process documents in batches asyncrhonously."""
    log_header("VECTOR STORAGE PHASE")
    log_info(
        f"  VectorStore Indexing: Preparing to add {len(documents)} documents to vector store",
        Colors.DARKCYAN,
    )

    batches = [
        documents[i : i + batch_size] for i in range(0, len(documents), batch_size)
    ]

    log_info(
        f"  VectorStore Indexing: Split into {len(batches)} batches of {batch_size} documents each"
    )

    # Process all batches concurrently
    async def add_batch(batch: List[Document], batch_num: int):
        try:
            # await chroma.add_documents(batch)
            await chroma.aadd_documents(batch)
            log_success(
                f"VectorStore Indexing: Successfully added batch {batch_num}/{len(batches)} ({len(batch)} documents)"
            )
        except Exception as e:
            log_error(f"VectorStore Indexing: Failed to add batch {batch_num} - {e}")
            return False

        return True

    # Processes batches concurrently
    tasks = [add_batch(batch, i + 1) for i, batch in enumerate(batches)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    successful = sum(1 for result in results if result is True)

    if successful == len(batches):
        log_success(
            f"VectorStore Indexing: All batches processed successfully! ({successful}/{len(batches)})"
        )
    else:
        log_warning(
            f"VectorStore Indexing: Processed {successful}/{len(batches)} batches successfully."
        )


async def main():
    """Main sync function to orchestrate the entire process."""
    log_header("DOCUMENTATION INGESTION PIPELINE")

    log_info(
        "   TavilyCrawl: Starting to Crawl documentation from https://python.langchain.com/",
        Colors.PURPLE,
    )

    # Crawl the site

    res = tavily_crawl.invoke(
        {
            "url": "https://python.langchain.com/",
            "max_depth": 2,
            "extract_depth": "advanced",
        }
    )

    all_docs = [
        Document(page_content=result["raw_content"], metadata={"source": result["url"]})
        for result in res["results"]
    ]
    log_success(
        f"TavilyCrawl: Successfully crawled {len(all_docs)} documents from https://python.langchain.com/"
    )

    log_header("DOCUMENT CHUNKING PHASE")

    log_info(
        f" Text Splitter: Processing {len(all_docs)} documents with 4000 chunk size and 200 overlap.",
        Colors.YELLOW,
    )
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(all_docs)
    log_success(
        f"Text Splitter: Created {len(split_docs)} from {len(all_docs)} documents."
    )

    await index_documents_async(split_docs, batch_size=500)

    log_header("PIPELINE COMPLETE")
    log_success(" Documentation ingestion pipeline finished successfully!")
    log_info(" Summary:", Colors.BOLD)
    log_info(f"    - URLs Mapped: {len(res['results'])}")
    log_info(f"    - Documents Extracted: {len(all_docs)}")
    log_info(f"    - Chunks Created: {len(split_docs)}")


if __name__ == "__main__":
    asyncio.run(main())
