# Step 1: Import necessary libraries
# --------------------------------------
# Import required libraries for document retrieval, reranking, and logging setup.
from sentence_transformers import CrossEncoder
import logging
import argparse
import torch
import numpy as np
import os
from rank_bm25 import BM25Okapi

from financerag.rerank import CrossEncoderReranker
from financerag.rerank.cross_encoder import format_qwen3_query, format_qwen3_document
from financerag.retrieval import DenseRetrieval, SentenceTransformerEncoder, BM25Retriever
from financerag.retrieval.bm25 import tokenize_list
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
parser.add_argument("--batch_size", type=int, default=4, help="Batch size for retrieval and reranking.")
parser.add_argument("--bm25_weight", type=float, default=0.5, help="Weight for BM25 scores in hybrid retrieval (0-1).")
parser.add_argument("--dense_weight", type=float, default=0.5, help="Weight for dense scores in hybrid retrieval (0-1).")
args = parser.parse_args()

# Step 2: Initialize Task
task = TASK[args.task]()


# Step 3: Initialize DenseRetriever model
# -------------------------------------
# Initialize the retrieval model using SentenceTransformers. This model will be responsible
# for encoding both the queries and documents into embeddings.
#
# You can replace 'intfloat/e5-large-v2' with any other model supported by SentenceTransformers.
# For example: 'BAAI/bge-large-en-v1.5', 'Linq-AI-Research/Linq-Embed-Mistral', etc.
encoder_model = SentenceTransformerEncoder(
    model_name_or_path='Qwen/Qwen3-Embedding-4B',
    query_prompt='query',
)

# Set max sequence length for Qwen3 models
'''if hasattr(encoder_model, 'q_model'):
    encoder_model.q_model.max_seq_length = 8192
    if hasattr(encoder_model, 'doc_model') and encoder_model.doc_model != encoder_model.q_model:
        encoder_model.doc_model.max_seq_length = 8192'''


# Step 4: Perform Dense Retrieval with Qwen3 Embeddings
# -------------------------------------------------------
# Use dense retrieval model to get embeddings and results
logging.info("Starting Dense Retrieval with Qwen3 Embeddings...")

retrieval_model = DenseRetrieval(
    model=encoder_model, 
    batch_size=args.batch_size
)

# Perform dense retrieval
dense_results = task.retrieve(
    retriever=retrieval_model,
    top_k=100,
)

# Step 5: Perform BM25 Retrieval (Lexical)
# -----------------------------------------
logging.info("Building BM25 index...")

# Prepare corpus for BM25
corpus_texts = []
corpus_ids_bm25 = []

for doc_id, doc in task.corpus.items():
    corpus_ids_bm25.append(doc_id)
    text = (doc.get('title', '') + ' ' + doc.get('text', '')).strip()
    corpus_texts.append(text)

# Tokenize corpus using the tokenize_list utility function
logging.info("Tokenizing corpus for BM25...")
tokenized_corpus = tokenize_list([doc.lower() for doc in corpus_texts])

# Build BM25 index
bm25_model = BM25Okapi(tokenized_corpus)
bm25_retriever = BM25Retriever(model=bm25_model)

# Perform BM25 retrieval
logging.info("Performing BM25 retrieval...")
bm25_results = task.retrieve(
    retriever=bm25_retriever,
    top_k=100,
)

# Step 6: Hybrid Retrieval - Combine BM25 and Dense Results
# ----------------------------------------------------------
logging.info(f"Combining BM25 (weight={args.bm25_weight}) and Dense (weight={args.dense_weight}) results...")

# Normalize scores for fair combination
def normalize_scores(results):
    """Min-max normalize scores per query"""
    normalized = {}
    for q_id, docs in results.items():
        if not docs:
            normalized[q_id] = {}
            continue
        scores = list(docs.values())
        min_score = min(scores)
        max_score = max(scores)
        score_range = max_score - min_score
        
        if score_range == 0:
            normalized[q_id] = {doc_id: 1.0 for doc_id in docs}
        else:
            normalized[q_id] = {
                doc_id: (score - min_score) / score_range 
                for doc_id, score in docs.items()
            }
    return normalized

# Normalize both result sets
bm25_normalized = normalize_scores(bm25_results)
dense_normalized = normalize_scores(dense_results)

# Combine scores
retrieval_result = {}
for q_id in task.queries.keys():
    retrieval_result[q_id] = {}
    
    # Get all unique document IDs from both methods
    bm25_docs = set(bm25_normalized.get(q_id, {}).keys())
    dense_docs = set(dense_normalized.get(q_id, {}).keys())
    all_docs = bm25_docs | dense_docs
    
    # Combine scores with weights
    for doc_id in all_docs:
        bm25_score = bm25_normalized.get(q_id, {}).get(doc_id, None)
        dense_score = dense_normalized.get(q_id, {}).get(doc_id, None)
        
        # Only include documents that appear in at least one result set
        # If doc not in a method, treat its contribution as 0 (but don't penalize)
        if bm25_score is None and dense_score is None:
            continue  # Skip docs not in either result set
        
        # Use 0.0 for missing scores, but only after confirming doc exists in at least one set
        bm25_contribution = args.bm25_weight * (bm25_score if bm25_score is not None else 0.0)
        dense_contribution = args.dense_weight * (dense_score if dense_score is not None else 0.0)
        
        combined_score = bm25_contribution + dense_contribution
        retrieval_result[q_id][doc_id] = combined_score

# Print a portion of the hybrid retrieval results to verify the output.
logging.info("=" * 80)
logging.info("HYBRID RETRIEVAL RESULTS (BM25 + Dense)")
logging.info("=" * 80)
print(f"\nHybrid retrieval results for {len(retrieval_result)} queries.")
print(f"Combination: BM25 (weight={args.bm25_weight}) + Dense (weight={args.dense_weight})")
print(f"\nTop 5 documents for the first query:")

for q_id, result in retrieval_result.items():
    print(f"\nQuery ID: {q_id}")
    print(f"Query: {task.queries[q_id][:100]}...")
    
    # Sort the result to print the top 5 document ID and its score
    sorted_results = sorted(result.items(), key=lambda x: x[1], reverse=True)

    for i, (doc_id, score) in enumerate(sorted_results[:5]):
        # Show which method retrieved this doc
        in_bm25 = doc_id in bm25_results.get(q_id, {})
        in_dense = doc_id in dense_results.get(q_id, {})
        source = []
        if in_bm25:
            source.append(f"BM25: {bm25_results[q_id][doc_id]:.4f}")
        if in_dense:
            source.append(f"Dense: {dense_results[q_id][doc_id]:.4f}")
        
        print(f"  {i + 1}. Document ID: {doc_id}")
        print(f"     Combined Score: {score:.4f}")
        print(f"     Sources: {', '.join(source)}")

    break  # Only show the first query


# Step 7: Initialize CrossEncoder Reranker
# --------------------------------------
# The CrossEncoder model will be used to rerank the retrieved documents based on relevance.
#
# You can replace 'cross-encoder/ms-marco-MiniLM-L-12-v2' with any other model supported by CrossEncoder.
# For example: 'cross-encoder/ms-marco-TinyBERT-L-2', 'cross-encoder/stsb-roberta-large', etc.
task_instruction = "Given a financial query, retrieve relevant financial documents that answer the query"

'''reranker = CrossEncoderReranker(
    model=CrossEncoder(
        "tomaarsen/Qwen3-Reranker-0.6B-seq-cls",
        trust_remote_code=True,
        num_labels=1,
        max_length=8192,),
    query_formatter=lambda q: format_qwen3_query(q, instruction=task_instruction),
    document_formatter=format_qwen3_document,
)'''
reranker = CrossEncoderReranker(
    model=CrossEncoder(
        "BAAI/bge-reranker-v2-m3",
        trust_remote_code=True,
        # num_labels=1,
        max_length=8192,)
)


# Step 8: Perform reranking
# -------------------------
# Rerank the top 100 documents from hybrid retrieval using the CrossEncoder model.
logging.info("Starting reranking of hybrid retrieval results...")
reranking_result = task.rerank(
    reranker=reranker,
    results=retrieval_result,
    top_k=100,  # Rerank the top 100 documents
    batch_size=args.batch_size
)

# Print a portion of the reranking results to verify the output.
logging.info("=" * 80)
logging.info("RERANKING RESULTS")
logging.info("=" * 80)
print(f"\nReranking results for {len(reranking_result)} queries.")
print(f"Top 5 documents for the first query after reranking:")

for q_id, result in reranking_result.items():
    print(f"\nQuery ID: {q_id}")
    print(f"Query: {task.queries[q_id][:100]}...")
    
    # Sort the result to print the top 5 document ID and its score
    sorted_results = sorted(result.items(), key=lambda x: x[1], reverse=True)

    for i, (doc_id, score) in enumerate(sorted_results[:5]):
        # Get original retrieval score
        hybrid_score = retrieval_result.get(q_id, {}).get(doc_id, 0.0)
        print(f"  {i + 1}. Document ID: {doc_id}")
        print(f"     Reranker Score: {score:.4f}")
        print(f"     Hybrid Retrieval Score: {hybrid_score:.4f}")

    break  # Only show the first query


# Step 9: Save results
# -------------------
# Save the results to the specified output directory as a CSV file.
output_dir = f'./results/qwen3_bm25_hybrid'
task.save_results(output_dir=output_dir)

# Confirm the results have been saved.
logging.info("=" * 80)
logging.info(f"Results saved to {output_dir}")
logging.info(f"Hybrid weights: BM25={args.bm25_weight}, Dense={args.dense_weight}")
logging.info("=" * 80)
print(f"\n✓ Pipeline completed successfully!")
print(f"✓ Results saved to: {output_dir}")
print(f"✓ Hybrid retrieval: BM25 ({args.bm25_weight}) + Dense ({args.dense_weight})")
