import heapq
import logging
from typing import Any, Callable, Dict, Literal, Optional

import torch
import numpy as np
from scipy.sparse import csr_matrix

from financerag.common.protocols import Retrieval

logger = logging.getLogger(__name__)


class HybridRetrieval(Retrieval):
    """
    Hybrid retrieval that combines dense and sparse representations.
    
    This class uses both dense embeddings (semantic similarity) and sparse embeddings 
    (lexical matching) to compute relevance scores between queries and documents.
    Scores from both methods are combined using weighted fusion.
    """

    def __init__(
            self,
            model,
            batch_size: int = 64,
            corpus_chunk_size: int = 50000,
            dense_weight: float = 0.5,
            sparse_weight: float = 0.5,
    ):
        """
        Initializes the HybridRetrieval class.

        Args:
            model: BGE-M3 model that supports both dense and sparse embeddings.
            batch_size (int, optional): Batch size for encoding. Defaults to 64.
            corpus_chunk_size (int, optional): Number of documents per batch. Defaults to 50000.
            dense_weight (float, optional): Weight for dense scores. Defaults to 0.5.
            sparse_weight (float, optional): Weight for sparse scores. Defaults to 0.5.
        """
        self.model = model
        self.batch_size = batch_size
        self.corpus_chunk_size = corpus_chunk_size
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.results: Dict = {}

    def retrieve(
            self,
            corpus: Dict[str, Dict[Literal["title", "text"], str]],
            queries: Dict[str, str],
            top_k: Optional[int] = None,
            return_sorted: bool = False,
            **kwargs,
    ) -> Dict[str, Dict[str, float]]:
        """
        Retrieves the top-k most relevant documents using hybrid search.

        Args:
            corpus: Dictionary of document IDs to document dicts.
            queries: Dictionary of query IDs to query texts.
            top_k: Number of top documents to return for each query.
            return_sorted: Whether to return results sorted by score.
            **kwargs: Additional arguments passed to the encoder model.

        Returns:
            Dictionary of query IDs to dictionaries of document IDs and scores.
        """
        logger.info("Encoding queries with dense and sparse representations...")
        query_ids = list(queries.keys())
        self.results = {qid: {} for qid in query_ids}
        query_texts = [queries[qid] for qid in queries]
        
        # Get both dense and sparse embeddings for queries
        query_embeddings = self.model.encode_queries(
            query_texts, batch_size=self.batch_size, return_dense=True, return_sparse=True, **kwargs
        )
        query_dense = query_embeddings['dense_vecs']
        query_sparse = query_embeddings['lexical_weights']

        logger.info("Sorting corpus by document length...")
        sorted_corpus_ids = sorted(
            corpus,
            key=lambda k: len(corpus[k].get("title", "") + corpus[k].get("text", "")),
            reverse=True,
        )

        logger.info("Encoding corpus in batches with dense and sparse representations...")
        result_heaps = {qid: [] for qid in query_ids}

        corpus_list = [corpus[cid] for cid in sorted_corpus_ids]

        for batch_num, start_idx in enumerate(
                range(0, len(corpus), self.corpus_chunk_size)
        ):
            logger.info(
                f"Encoding batch {batch_num + 1}/{len(range(0, len(corpus_list), self.corpus_chunk_size))}..."
            )
            end_idx = min(start_idx + self.corpus_chunk_size, len(corpus_list))

            # Encode chunk of corpus with both dense and sparse
            corpus_embeddings = self.model.encode_corpus(
                corpus_list[start_idx:end_idx], 
                batch_size=self.batch_size,
                return_dense=True,
                return_sparse=True,
                **kwargs
            )
            corpus_dense = corpus_embeddings['dense_vecs']
            corpus_sparse = corpus_embeddings['lexical_weights']

            # Compute dense scores (cosine similarity)
            dense_scores = self._compute_dense_scores(query_dense, corpus_dense)
            
            # Compute sparse scores (lexical matching)
            sparse_scores = self._compute_sparse_scores(query_sparse, corpus_sparse)
            
            # Normalize scores to [0, 1] range
            dense_scores_norm = self._normalize_scores(dense_scores)
            sparse_scores_norm = self._normalize_scores(sparse_scores)
            
            # Combine scores with weights
            combined_scores = (
                self.dense_weight * dense_scores_norm + 
                self.sparse_weight * sparse_scores_norm
            )
            
            combined_scores[torch.isnan(combined_scores)] = -1

            # Get top-k values
            if top_k is None:
                top_k = len(combined_scores[0])

            scores_top_k_values, scores_top_k_idx = torch.topk(
                combined_scores,
                min(top_k + 1, len(combined_scores[0])),
                dim=1,
                largest=True,
                sorted=return_sorted,
            )

            scores_top_k_values = scores_top_k_values.cpu().tolist()
            scores_top_k_idx = scores_top_k_idx.cpu().tolist()

            for query_itr in range(len(query_dense)):
                query_id = query_ids[query_itr]
                for sub_corpus_id, score in zip(
                        scores_top_k_idx[query_itr], scores_top_k_values[query_itr]
                ):
                    corpus_id = sorted_corpus_ids[start_idx + sub_corpus_id]
                    if corpus_id != query_id:
                        if len(result_heaps[query_id]) < top_k:
                            heapq.heappush(result_heaps[query_id], (score, corpus_id))
                        else:
                            heapq.heappushpop(
                                result_heaps[query_id], (score, corpus_id)
                            )

        for qid in result_heaps:
            for score, corpus_id in result_heaps[qid]:
                self.results[qid][corpus_id] = score

        return self.results

    def _compute_dense_scores(self, query_embeddings: torch.Tensor, corpus_embeddings: torch.Tensor) -> torch.Tensor:
        """Compute cosine similarity between query and corpus dense embeddings."""
        query_embeddings = self._ensure_tensor(query_embeddings)
        corpus_embeddings = self._ensure_tensor(corpus_embeddings)
        
        return torch.mm(
            torch.nn.functional.normalize(query_embeddings, p=2, dim=1),
            torch.nn.functional.normalize(corpus_embeddings, p=2, dim=1).transpose(0, 1),
        )

    def _compute_sparse_scores(self, query_sparse: list, corpus_sparse: list) -> torch.Tensor:
        """
        Compute sparse scores using lexical weights (like BM25-style matching).
        
        Args:
            query_sparse: List of dicts, where each dict maps token IDs to weights for a query.
            corpus_sparse: List of dicts, where each dict maps token IDs to weights for a document.
        
        Returns:
            Tensor of shape (num_queries, num_docs) with sparse matching scores.
        """
        num_queries = len(query_sparse)
        num_docs = len(corpus_sparse)
        
        scores = torch.zeros(num_queries, num_docs)
        
        for i, q_sparse in enumerate(query_sparse):
            for j, d_sparse in enumerate(corpus_sparse):
                # Compute dot product of sparse vectors
                score = 0.0
                if isinstance(q_sparse, dict) and isinstance(d_sparse, dict):
                    for token_id, q_weight in q_sparse.items():
                        if token_id in d_sparse:
                            score += q_weight * d_sparse[token_id]
                scores[i, j] = score
        
        return scores

    def _normalize_scores(self, scores: torch.Tensor) -> torch.Tensor:
        """Normalize scores to [0, 1] range using min-max normalization per query."""
        normalized = torch.zeros_like(scores)
        for i in range(scores.shape[0]):
            row = scores[i]
            min_val = row.min()
            max_val = row.max()
            if max_val > min_val:
                normalized[i] = (row - min_val) / (max_val - min_val)
            else:
                normalized[i] = row
        return normalized

    def _ensure_tensor(self, x: Any) -> torch.Tensor:
        """Ensures the input is a torch.Tensor, converting if necessary."""
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        return x
