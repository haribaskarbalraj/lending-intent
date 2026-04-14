"""
RAG vector store — dual implementation.

  app_env=dev  → ChromaRAGStore  (in-memory ChromaDB, no infra needed)
  app_env=prod → OpenSearchRAGStore (AWS OpenSearch Service k-NN index)

Spring analogy: this is your @Repository with two implementations —
like having an H2Repository for tests and a JpaRepository for prod,
both behind the same interface, swapped via @Profile.
"""

import logging
from abc import ABC, abstractmethod
from functools import lru_cache

from src.core.config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract interface — every store must implement these three methods
# ---------------------------------------------------------------------------

class RAGStore(ABC):
    """Common interface for all vector store backends.

    Spring analogy: this is the Repository interface that both
    ChromaRAGStore and OpenSearchRAGStore implement.
    """

    @abstractmethod
    def store(self, narrative: str, customer_id: str, metadata: dict) -> None:
        """Embed and persist a converter narrative."""

    @abstractmethod
    def retrieve_similar(self, narrative: str, n_results: int = 3) -> list[str]:
        """Return the top-N most similar past converter narratives."""

    @abstractmethod
    def count(self) -> int:
        """Return total number of stored narratives."""


# ---------------------------------------------------------------------------
# Dev implementation — ChromaDB in-memory (no AWS needed)
# ---------------------------------------------------------------------------

class ChromaRAGStore(RAGStore):
    """In-memory ChromaDB store for local development.

    Uses ONNX-based sentence embeddings that run locally —
    no API key or network call required.

    Spring analogy: the H2 / in-memory DataSource profile.
    """

    def __init__(self) -> None:
        # Import here so the prod path never loads chromadb unnecessarily
        from chromadb import EphemeralClient
        from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

        self._collection = (
            EphemeralClient()
            .get_or_create_collection(
                name="converter_narratives",
                embedding_function=ONNXMiniLM_L6_V2(),
            )
        )
        logger.info("ChromaRAGStore initialised (in-memory)")

    def store(self, narrative: str, customer_id: str, metadata: dict) -> None:
        self._collection.upsert(
            documents=[narrative],
            ids=[customer_id],
            metadatas=[metadata],
        )
        logger.debug("ChromaRAGStore: stored customer_id=%s", customer_id)

    def retrieve_similar(self, narrative: str, n_results: int = 3) -> list[str]:
        total = self._collection.count()
        if total == 0:
            logger.debug("ChromaRAGStore: collection is empty, skipping RAG")
            return []
        results = self._collection.query(
            query_texts=[narrative],
            n_results=min(n_results, total),
        )
        docs: list[str] = results["documents"][0]
        logger.debug("ChromaRAGStore: retrieved %d similar narratives", len(docs))
        return docs

    def count(self) -> int:
        return self._collection.count()


# ---------------------------------------------------------------------------
# Prod implementation — AWS OpenSearch Service k-NN index
# ---------------------------------------------------------------------------

class OpenSearchRAGStore(RAGStore):
    """AWS OpenSearch Service vector store for production.

    Uses Bedrock Titan Embeddings to embed narratives before indexing.
    Requires OPENSEARCH_HOST, OPENSEARCH_USERNAME, OPENSEARCH_PASSWORD
    in your .env.prod file.

    Spring analogy: the JPA / RDS DataSource profile — same interface,
    fully managed persistent storage.
    """

    # k-NN index settings — Titan v2 produces 1024-dim vectors
    _VECTOR_DIM = 1024
    _INDEX_MAPPING = {
        "settings": {
            "index": {
                "knn": True,
                "knn.algo_param.ef_search": 100,
            }
        },
        "mappings": {
            "properties": {
                "vector_field": {
                    "type": "knn_vector",
                    "dimension": _VECTOR_DIM,
                    "method": {
                        "name": "hnsw",
                        "space_type": "l2",
                        "engine": "nmslib",
                    },
                },
                "text": {"type": "text"},
                "metadata": {"type": "object"},
            }
        },
    }

    def __init__(self) -> None:
        # Lazy imports — only loaded when app_env=prod
        from langchain_aws import BedrockEmbeddings
        from langchain_community.vectorstores import OpenSearchVectorSearch
        from opensearchpy import OpenSearch, RequestsHttpConnection

        if not settings.opensearch_host:
            raise ValueError(
                "OPENSEARCH_HOST must be set in .env.prod for prod RAG store"
            )

        self._embeddings = BedrockEmbeddings(
            model_id=settings.embedding_model_id,
            region_name=settings.aws_region,
        )

        self._store = OpenSearchVectorSearch(
            opensearch_url=f"https://{settings.opensearch_host}:{settings.opensearch_port}",
            index_name=settings.opensearch_index,
            embedding_function=self._embeddings,
            http_auth=(settings.opensearch_username, settings.opensearch_password),
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection,
        )
        logger.info(
            "OpenSearchRAGStore initialised (host=%s, index=%s)",
            settings.opensearch_host,
            settings.opensearch_index,
        )

    def store(self, narrative: str, customer_id: str, metadata: dict) -> None:
        from langchain_core.documents import Document

        doc = Document(page_content=narrative, metadata={**metadata, "customer_id": customer_id})
        self._store.add_documents([doc], ids=[customer_id])
        logger.debug("OpenSearchRAGStore: stored customer_id=%s", customer_id)

    def retrieve_similar(self, narrative: str, n_results: int = 3) -> list[str]:
        results = self._store.similarity_search(narrative, k=n_results)
        docs = [doc.page_content for doc in results]
        logger.debug("OpenSearchRAGStore: retrieved %d similar narratives", len(docs))
        return docs

    def count(self) -> int:
        # OpenSearch doesn't expose a cheap count via LangChain wrapper;
        # return a sentinel so callers know the store is live
        return -1


# ---------------------------------------------------------------------------
# Factory — mirrors get_llm_client() pattern already in the codebase
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_rag_store() -> RAGStore:
    """Return the correct RAGStore based on APP_ENV.

    lru_cache makes this a singleton — same pattern as get_llm_client().
    Spring analogy: @Bean with @Profile("dev") / @Profile("prod").
    """
    if settings.app_env == "prod":
        logger.info("RAG backend: OpenSearch (prod)")
        return OpenSearchRAGStore()
    logger.info("RAG backend: ChromaDB in-memory (dev)")
    return ChromaRAGStore()
