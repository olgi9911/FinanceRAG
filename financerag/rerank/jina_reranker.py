import logging
from typing import Dict, Optional, List
import torch
from transformers import AutoModel, AutoTokenizer
from financerag.common import Reranker
from tqdm import tqdm

logger = logging.getLogger(__name__)

class JinaListwiseReranker(Reranker):
    def __init__(
        self, 
        model_name: str = "jinaai/jina-reranker-v3",
        device: str = "cuda",
        token: Optional[str] = None,
        default_batch_size: int = 16, # Documents per attention pass
        max_total_length: Optional[int] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.default_batch_size = default_batch_size
        self.max_total_length = max_total_length

        logger.info(f"Loading Jina Listwise Reranker: {model_name} on {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, token=token
        )
        self.model = AutoModel.from_pretrained(
            model_name, 
            trust_remote_code=True, 
            torch_dtype=torch.float16 if "cuda" in self.device else torch.float32,
            token=token
        ).to(self.device).eval()
        
        self.results: Dict[str, Dict[str, float]] = {}

    def rerank(
            self,
            corpus: Dict[str, Dict[str, str]],
            queries: Dict[str, str],
            results: Dict[str, Dict[str, float]],
            top_k: int,
            batch_size: Optional[int] = None, 
            **kwargs
    ) -> Dict[str, Dict[str, float]]:
        
        # 1. Determine Micro-Batch Size
        # This controls how many docs are sent to Jina in one call
        micro_batch_size = batch_size if batch_size is not None else self.default_batch_size

        self.results = {query_id: {} for query_id in results}
        logger.info(f"Starting Jina Reranking with micro_batch_size={micro_batch_size}")

        for query_id in tqdm(results, desc="Reranking Queries"):
            # --- Candidate Selection ---
            if len(results[query_id]) > top_k:
                candidates = sorted(
                    results[query_id].items(), key=lambda item: item[1], reverse=True
                )[:top_k]
            else:
                candidates = list(results[query_id].items())

            if not candidates:
                continue

            doc_ids_all = [doc_id for doc_id, _ in candidates]
            query_text = queries[query_id]
            
            # Prepare all document texts
            all_docs_text = [
                (corpus[did].get("title", "") + " " + corpus[did].get("text", "")).strip()
                for did in doc_ids_all
            ]

            # --- MANUAL BATCHING LOOP ---
            # We must manually slice the list because model.rerank() doesn't do it for us.
            for i in range(0, len(all_docs_text), micro_batch_size):
                
                # Slice the batch
                batch_docs_text = all_docs_text[i : i + micro_batch_size]
                batch_doc_ids = doc_ids_all[i : i + micro_batch_size]

                # Call Jina on this specific chunk
                # Note: We removed 'batch_size' from arguments here
                jina_output = self.model.rerank(
                    query=query_text,
                    documents=batch_docs_text,
                    # max_query_length=kwargs.get('max_query_length', 512),
                    # max_length=self.max_total_length
                )

                # Process results for this chunk
                for item in jina_output:
                    # 'index' is relative to the current batch (0 to micro_batch_size-1)
                    relative_index = item['index']
                    score = float(item['relevance_score'])
                    
                    # Map back to the correct Doc ID
                    doc_id = batch_doc_ids[relative_index]
                    self.results[query_id][doc_id] = score

        return self.results