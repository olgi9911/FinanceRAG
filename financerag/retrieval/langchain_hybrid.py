"""
LangChain-based hybrid retrieval for FinanceRAG.

This module provides a wrapper around LangChain's EnsembleRetriever that combines:
- BM25Retriever (lexical/sparse retrieval)
- Vector retriever (dense/semantic retrieval using embeddings)

The wrapper maintains compatibility with the FinanceRAG pipeline interface.
"""

import logging
from typing import Dict, List, Literal, Optional
import numpy as np

try:
    from langchain_classic.retrievers import EnsembleRetriever
    from langchain_community.retrievers import BM25Retriever
    from langchain_core.embeddings import Embeddings
    from langchain_community.vectorstores import Chroma
    from langchain_core.documents import Document
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

from financerag.common.protocols import Encoder, Retrieval

logger = logging.getLogger(__name__)


class LangChainHybridRetrieval(Retrieval):
    """
    Hybrid retrieval using LangChain's EnsembleRetriever.
    
    This class wraps LangChain's EnsembleRetriever to combine BM25 and dense
    retrieval, while maintaining compatibility with the FinanceRAG pipeline.
    
    Features:
    - Uses LangChain's BM25Retriever for lexical matching
    - Uses ChromaDB for dense vector retrieval
    - Reciprocal Rank Fusion (RRF) for combining results
    - Persistent ChromaDB storage (optional)
    
    Example:
        >>> from financerag.retrieval import LangChainHybridRetrieval, SentenceTransformerEncoder
        >>> encoder = SentenceTransformerEncoder("Qwen/Qwen3-Embedding-4B")
        >>> retriever = LangChainHybridRetrieval(
        ...     encoder_model=encoder,
        ...     bm25_weight=0.5,
        ...     dense_weight=0.5,
        ...     persist_directory="./chromadb"
        ... )
        >>> results = task.retrieve(retriever, top_k=100)
    """
    
    def __init__(
        self,
        encoder_model: Optional[Encoder] = None,
        embeddings: Optional[Embeddings] = None,
        bm25_weight: float = 0.5,
        dense_weight: float = 0.5,
        persist_directory: Optional[str] = "./chromadb",
        collection_name: str = "finance_rag",  # Should include task name, e.g., "finance_rag_finqa"
        batch_size: int = 64,
    ):
        """
        Initialize LangChain hybrid retriever.
        
        Args:
            encoder_model: Encoder implementing the Encoder protocol
            bm25_weight: Weight for BM25 retriever (0-1)
            dense_weight: Weight for dense retriever (0-1)
            persist_directory: Directory for ChromaDB persistence (None for in-memory)
            collection_name: Name for ChromaDB collection (include task name, e.g., "finance_rag_finqa")
            batch_size: Batch size for encoding
        
        Raises:
            ImportError: If langchain or required packages not installed
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "LangChain is not installed. Install with:\n"
                "pip install langchain langchain-community chromadb"
            )
        
        self.encoder_model = encoder_model
        self.embeddings = embeddings
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.batch_size = batch_size
        
        # Normalize weights
        total_weight = bm25_weight + dense_weight
        if abs(total_weight - 1.0) > 0.01:
            logger.warning(
                f"Weights don't sum to 1.0 ({total_weight}). "
                f"Normalizing: BM25={bm25_weight/total_weight:.3f}, "
                f"Dense={dense_weight/total_weight:.3f}"
            )
            self.bm25_weight = bm25_weight / total_weight
            self.dense_weight = dense_weight / total_weight
        
        # Will be initialized in retrieve()
        self.ensemble_retriever = None
        self.corpus_ids = []
        self._embeddings_wrapper = None
    
    def _create_embeddings_wrapper(self):
        """Create LangChain Embeddings wrapper for our encoder."""
        encoder = self.encoder_model
        batch_size = self.batch_size
        
        class FinanceRAGEmbeddings(Embeddings):
            """Wrapper to make FinanceRAG encoder compatible with LangChain.
            
            Reuses sent_encoder.py's encode_corpus/encode_queries for consistency.
            """
            
            def embed_documents(self, texts: List[str]) -> List[List[float]]:
                """Embed a list of documents.
                
                Note: texts are already title+text combined from LangChain Documents.
                We pass them with empty title to match sent_encoder.py's expected format.
                """
                # Convert to corpus format matching sent_encoder.py
                # Since texts are already combined, we set title="" and text=combined_text
                # This aligns with sent_encoder.py which will output: text.strip()
                corpus = [{"title": "", "text": text} for text in texts]
                embeddings = encoder.encode_corpus(corpus, batch_size=batch_size)
                
                # Convert to list of lists
                if hasattr(embeddings, 'cpu'):
                    embeddings = embeddings.cpu().numpy()
                embeddings = np.array(embeddings, dtype=np.float32)
                
                return embeddings.tolist()
            
            def embed_query(self, text: str) -> List[float]:
                """Embed a single query."""
                # Directly use encoder's encode_queries (handles prompts internally)
                embedding = encoder.encode_queries(
                    [text], 
                    batch_size=1,
                    show_progress_bar=False  # Disable progress bars for cleaner output
                )
                
                # Convert to list
                if hasattr(embedding, 'cpu'):
                    embedding = embedding.cpu().numpy()
                embedding = np.array(embedding, dtype=np.float32)
                
                return embedding[0].tolist()
        
        return FinanceRAGEmbeddings()
    
    def _build_ensemble_retriever(
        self,
        corpus: Dict[str, Dict[Literal["title", "text"], str]],
        top_k: int
    ):
        """
        Build LangChain EnsembleRetriever from corpus.
        
        Args:
            corpus: Dictionary mapping doc_ids to {title, text}
            top_k: Number of documents to retrieve
        """
        logger.info("Building LangChain EnsembleRetriever...")
        
        # Prepare documents for LangChain
        documents = []
        self.corpus_ids = []
        
        for doc_id, doc in corpus.items():
            # Combine title and text (align with sent_encoder.py)
            if doc.get("title", ""):
                text = (doc["title"] + " " + doc["text"]).strip()
            else:
                text = doc["text"].strip()
            
            # Create LangChain Document with metadata
            lc_doc = Document(
                page_content=text,
                metadata={"doc_id": doc_id}
            )
            documents.append(lc_doc)
            self.corpus_ids.append(doc_id)
        
        logger.info(f"Prepared {len(documents)} documents for LangChain")
        
        # 1. Create BM25 retriever
        logger.info("Building BM25 retriever...")
        bm25_retriever = BM25Retriever.from_documents(documents)
        bm25_retriever.k = top_k
        logger.info(f"BM25 retriever built with k={top_k}")
        
        # 2. Create Dense retriever using ChromaDB
        logger.info("Building dense retriever with ChromaDB...")
        
        if self.embeddings:
            embedding_function = self.embeddings
        else:
            if self._embeddings_wrapper is None:
                self._embeddings_wrapper = self._create_embeddings_wrapper()
            embedding_function = self._embeddings_wrapper
        
        # Build ChromaDB vector store
        if self.persist_directory:
            logger.info(f"Using persistent ChromaDB at: {self.persist_directory}")
            
            # Try to load existing collection first
            try:
                import chromadb
                client = chromadb.PersistentClient(path=self.persist_directory)
                
                # Check if collection exists
                existing_collections = [col.name for col in client.list_collections()]
                
                if self.collection_name in existing_collections:
                    logger.info(f"Loading existing collection: {self.collection_name}")
                    vectorstore = Chroma(
                        client=client,
                        collection_name=self.collection_name,
                        embedding_function=embedding_function
                    )
                else:
                    logger.info(f"Creating new collection: {self.collection_name}")
                    vectorstore = Chroma.from_documents(
                        documents=documents,
                        embedding=embedding_function,
                        client=client,
                        collection_name=self.collection_name
                    )
            except Exception as e:
                logger.warning(f"Failed to use persistent client: {e}. Falling back to from_documents().")
                vectorstore = Chroma.from_documents(
                    documents=documents,
                    embedding=embedding_function,
                    persist_directory=self.persist_directory,
                    collection_name=self.collection_name
                )
        else:
            logger.info("Using in-memory ChromaDB (no persistence)")
            vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=embedding_function,
                collection_name=self.collection_name
            )
        
        dense_retriever = vectorstore.as_retriever(
            search_kwargs={"k": top_k}
        )
        logger.info(f"Dense retriever built with k={top_k}")
        
        # 3. Create Ensemble retriever with RRF
        logger.info(f"Creating EnsembleRetriever with weights: BM25={self.bm25_weight}, Dense={self.dense_weight}")
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, dense_retriever],
            weights=[self.bm25_weight, self.dense_weight]
        )
        
        logger.info("EnsembleRetriever built successfully!")
    
    def retrieve(
        self,
        corpus: Dict[str, Dict[Literal["title", "text"], str]],
        queries: Dict[str, str],
        top_k: Optional[int] = None,
        score_function: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Dict[str, float]]:
        """
        Retrieve relevant documents using LangChain EnsembleRetriever.
        
        Args:
            corpus: Dictionary mapping doc_ids to {title, text}
            queries: Dictionary mapping query_ids to query text
            top_k: Number of top documents to retrieve per query
            score_function: Not used (LangChain handles scoring)
            **kwargs: Additional arguments
        
        Returns:
            Dictionary mapping query_ids to {doc_id: score}
        """
        if top_k is None:
            top_k = 100
        
        # Build ensemble retriever if not exists
        if self.ensemble_retriever is None:
            self._build_ensemble_retriever(corpus, top_k)
        
        # Retrieve for each query
        results = {}
        query_ids = list(queries.keys())
        
        logger.info(f"Retrieving top-{top_k} documents for {len(queries)} queries...")
        
        for i, (qid, query_text) in enumerate(queries.items()):
            # Use LangChain ensemble retriever
            retrieved_docs = self.ensemble_retriever.invoke(query_text)
            
            # Convert to FinanceRAG format: {doc_id: score}
            query_results = {}
            
            for rank, doc in enumerate(retrieved_docs[:top_k]):
                doc_id = doc.metadata.get("doc_id")
                if doc_id:
                    # Use reciprocal rank as score (since LangChain uses RRF internally)
                    score = 1.0 / (rank + 1)
                    
                    # Aggregate if same doc appears multiple times (max pooling)
                    if doc_id in query_results:
                        query_results[doc_id] = max(query_results[doc_id], score)
                    else:
                        query_results[doc_id] = score
            
            results[qid] = query_results
            
            if (i + 1) % 50 == 0:
                logger.info(f"Processed {i + 1}/{len(queries)} queries")
        
        logger.info(f"Retrieval complete for {len(results)} queries")
        return results
