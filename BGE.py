# Step 1: Import necessary libraries
# --------------------------------------
# Import required libraries for document retrieval, reranking, and logging setup.
from FlagEmbedding import BGEM3FlagModel
import logging
import numpy as np
import argparse

from financerag.retrieval import DenseRetrieval
from financerag.tasks import ConvFinQA, FinanceBench, FinDER, FinQA, FinQABench, MultiHiertt, TATQA

TASK = {
    "ConvFinQA": ConvFinQA,
    "FinanceBench": FinanceBench,
    "FinDER": FinDER,
    "FinQA": FinQA,
    "FinQABench": FinQABench,
    "MultiHiertt": MultiHiertt,
    "TATQA": TATQA,
}

# Setup basic logging configuration to show info level messages.
logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser(description="Run FinanceRAG baseline with different tasks.")
parser.add_argument(
    "--task",
    type=str,
    choices=TASK.keys(),
    default="FinDER",
    help="Specify the task to run."
)
args = parser.parse_args()

# Step 2: Initialize Task
task = TASK[args.task]()


# Step 3: Initialize BGE-M3 model wrapper
# -------------------------------------
# Create a wrapper class for BGE-M3 model to make it compatible with the DenseRetrieval interface.
class BGEM3Encoder:
    def __init__(self, model_name_or_path='BAAI/bge-m3'):
        """Initialize BGE-M3 model for encoding queries and documents."""
        import torch
        
        # Check if CUDA is available
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Initialize BGE-M3 model
        # The model uses safetensors format by default, which is safe
        self.model = BGEM3FlagModel(
            model_name_or_path,
            use_fp16=True,
            device=device
        )
    
    def encode_queries(self, queries, batch_size=32, **kwargs):
        """Encode queries using BGE-M3 model."""
        # Handle different input formats (list of strings or list of dicts)
        if isinstance(queries, list) and len(queries) > 0:
            if isinstance(queries[0], dict):
                queries = [q.get('text', str(q)) for q in queries]
        
        embeddings = self.model.encode(
            queries,
            batch_size=batch_size,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False
        )['dense_vecs']
        return embeddings
    
    def encode_corpus(self, corpus, batch_size=32, **kwargs):
        """Encode corpus documents using BGE-M3 model."""
        # Handle different input formats (list of strings or list of dicts)
        if isinstance(corpus, list) and len(corpus) > 0:
            if isinstance(corpus[0], dict):
                # Extract text from dict, preferring 'text' field or concatenating title+text
                corpus = [
                    (doc.get('title', '') + ' ' + doc.get('text', '')).strip() if 'title' in doc or 'text' in doc
                    else str(doc)
                    for doc in corpus
                ]
        
        embeddings = self.model.encode(
            corpus,
            batch_size=batch_size,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False
        )['dense_vecs']
        return embeddings


# Step 4: Initialize BGE-M3 Retriever
# ------------------------------------
# Initialize the retrieval model using BGE-M3. This model will be responsible
# for encoding both the queries and documents into embeddings.
encoder_model = BGEM3Encoder(model_name_or_path='BAAI/bge-m3')

retrieval_model = DenseRetrieval(
    model=encoder_model
)


# Step 5: Perform retrieval
# ---------------------
# Use the model to retrieve relevant documents for given queries.

retrieval_result = task.retrieve(
    retriever=retrieval_model
)

# Print a portion of the retrieval results to verify the output.
print(f"Retrieved results for {len(retrieval_result)} queries. Here's an example of the top 5 documents for the first query:")

for q_id, result in retrieval_result.items():
    print(f"\nQuery ID: {q_id}")
    # Sort the result to print the top 5 document ID and its score
    sorted_results = sorted(result.items(), key=lambda x: x[1], reverse=True)

    for i, (doc_id, score) in enumerate(sorted_results[:5]):
        print(f"  Document {i + 1}: Document ID = {doc_id}, Score = {score}")

    break  # Only show the first query


# Step 6: Initialize BGE Reranker v2-m3
# --------------------------------------
# Import the base Reranker class and FlagReranker
from financerag.common import Reranker
from FlagEmbedding import FlagReranker

# Create a reranker using BGE-reranker-v2-m3 model for reranking retrieved documents.
class BGERerankerV2M3(Reranker):
    """
    A reranker class that utilizes BGE-reranker-v2-m3 model to rerank search results.
    This model is specifically designed for reranking tasks.
    """
    
    def __init__(self, model_name_or_path='BAAI/bge-reranker-v2-m3'):
        """Initialize BGE reranker v2-m3."""
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = FlagReranker(model_name_or_path, use_fp16=True, device=device)
        self.results = {}
    
    def rerank(
            self,
            corpus,
            queries,
            results,
            top_k,
            batch_size=None,
            **kwargs
    ):
        """
        Reranks the top-k documents for each query using BGE-M3 model.
        
        Args:
            corpus: Dictionary of document IDs to document dicts (with 'title' and 'text')
            queries: Dictionary of query IDs to query texts
            results: Dictionary of query IDs to dictionaries of document IDs and scores
            top_k: Number of top documents to rerank for each query
            batch_size: Batch size for processing
            
        Returns:
            Dictionary of query IDs to dictionaries of reranked document IDs and scores
        """
        import logging
        logger = logging.getLogger(__name__)
        
        sentence_pairs, pair_ids = [], []
        
        # Prepare query-document pairs for reranking
        for query_id in results:
            if len(results[query_id]) > top_k:
                # Only rerank top-k documents
                for doc_id, _ in sorted(
                    results[query_id].items(), key=lambda item: item[1], reverse=True
                )[:top_k]:
                    pair_ids.append([query_id, doc_id])
                    corpus_text = (
                        corpus[doc_id].get("title", "")
                        + " "
                        + corpus[doc_id].get("text", "")
                    ).strip()
                    sentence_pairs.append([queries[query_id], corpus_text])
            else:
                # Rerank all documents
                for doc_id in results[query_id]:
                    pair_ids.append([query_id, doc_id])
                    corpus_text = (
                        corpus[doc_id].get("title", "")
                        + " "
                        + corpus[doc_id].get("text", "")
                    ).strip()
                    sentence_pairs.append([queries[query_id], corpus_text])
        
        # Perform reranking using BGE-reranker-v2-m3
        logger.info(f"Starting To Rerank Top-{top_k} using BGE-reranker-v2-m3....")
        
        # FlagReranker.compute_score expects pairs and returns scores directly
        rerank_scores = self.model.compute_score(
            sentence_pairs,
            batch_size=batch_size if batch_size else 32
        )
        
        # Ensure it's a list of floats
        if not isinstance(rerank_scores, list):
            rerank_scores = [float(rerank_scores)]
        else:
            rerank_scores = [float(score) for score in rerank_scores]
        
        # Organize reranked results
        self.results = {query_id: {} for query_id in results}
        for pair, score in zip(pair_ids, rerank_scores):
            query_id, doc_id = pair[0], pair[1]
            self.results[query_id][doc_id] = score
        
        return self.results


# Initialize BGE reranker v2-m3 model
reranker = BGERerankerV2M3(model_name_or_path='BAAI/bge-reranker-v2-m3')


# Step 7: Perform reranking
# -------------------------
# Rerank the top 100 retrieved documents using the BGE-reranker-v2-m3 model.
reranking_result = task.rerank(
    reranker=reranker,
    results=retrieval_result,
    top_k=100,  # Rerank the top 100 documents
    batch_size=32
)

# Print a portion of the reranking results to verify the output.
print(f"Reranking results for {len(reranking_result)} queries. Here's an example of the top 5 documents for the first query:")

for q_id, result in reranking_result.items():
    print(f"\nQuery ID: {q_id}")
    # Sort the result to print the top 5 document ID and its score
    sorted_results = sorted(result.items(), key=lambda x: x[1], reverse=True)

    for i, (doc_id, score) in enumerate(sorted_results[:5]):
        print(f"  Document {i + 1}: Document ID = {doc_id}, Score = {score}")

    break  # Only show the first query


# Step 8: Save results
# -------------------
# Save the results to the specified output directory as a CSV file.
output_dir = './results/baseline'
task.save_results(output_dir=output_dir)

# Confirm the results have been saved.
print("Results have been saved to results.csv")