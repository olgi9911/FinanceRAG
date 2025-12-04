"""
BM25-Only Retrieval Pipeline for FinanceRAG

This script performs retrieval using only BM25 (lexical/keyword-based matching).
No dense embeddings, just fast BM25 retrieval followed by optional reranking.

Usage:
    python bm25.py --task FinQA
    python bm25.py --task FinanceBench --retrieval_top_k 200 --rerank_top_k 100
"""

import logging
import argparse
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from financerag.retrieval import BM25Retriever
from financerag.retrieval.bm25 import tokenize_list
from financerag.rerank import CrossEncoderReranker
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

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Parse arguments
parser = argparse.ArgumentParser(description="Run FinanceRAG with BM25-only retrieval")
parser.add_argument(
    "--task",
    type=str,
    choices=TASK.keys(),
    default="FinQA",
    help="Specify the task to run"
)
parser.add_argument(
    "--retrieval_top_k",
    type=int,
    default=None,
    help="Number of documents to retrieve with BM25"
)
parser.add_argument(
    "--rerank_top_k",
    type=int,
    default=100,
    help="Number of documents to rerank with CrossEncoder"
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=4,
    help="Batch size for reranking"
)
parser.add_argument(
    "--no_rerank",
    action="store_true",
    help="Skip reranking step (BM25 results only)"
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="./results/bm25",
    help="Directory to save results"
)

args = parser.parse_args()

logger.info("=" * 80)
logger.info("FinanceRAG - BM25 Only Retrieval")
logger.info("=" * 80)
logger.info(f"Task: {args.task}")
logger.info(f"Retrieval Top-K: {args.retrieval_top_k}")
logger.info(f"Rerank: {'No' if args.no_rerank else f'Yes (top-{args.rerank_top_k})'}")
logger.info("=" * 80)

# ============================================================================
# Step 1: Initialize Task
# ============================================================================
logger.info("\n[Step 1/3] Initializing task...")
task = TASK[args.task]()
logger.info(f"✓ Loaded {len(task.corpus)} documents and {len(task.queries)} queries")

# ============================================================================
# Step 2: BM25 Retrieval
# ============================================================================
logger.info("\n[Step 2/3] Building BM25 index and retrieving...")

# Prepare corpus for BM25
corpus_texts = []
corpus_ids = []

for doc_id, doc in task.corpus.items():
    corpus_ids.append(doc_id)
    text = (doc.get('title', '') + ' ' + doc.get('text', '')).strip()
    corpus_texts.append(text)

# Tokenize corpus
logger.info("Tokenizing corpus...")
tokenized_corpus = tokenize_list([doc.lower() for doc in corpus_texts])

# Build BM25 index
logger.info("Building BM25 index...")
bm25_model = BM25Okapi(tokenized_corpus)

# Create BM25 retriever
bm25_retriever = BM25Retriever(model=bm25_model)

# Retrieve
logger.info(f"Retrieving top-{args.retrieval_top_k} documents per query...")
retrieval_results = task.retrieve(
    retriever=bm25_retriever,
    top_k=args.retrieval_top_k
)

# Log statistics
logger.info("\n--- BM25 Retrieval Statistics ---")
logger.info(f"Total queries processed: {len(retrieval_results)}")

if retrieval_results:
    sample_query = list(retrieval_results.keys())[0]
    logger.info(f"Sample query: {sample_query}")
    logger.info(f"  Retrieved {len(retrieval_results[sample_query])} documents")
    
    top_3_docs = sorted(
        retrieval_results[sample_query].items(),
        key=lambda x: x[1],
        reverse=True
    )[:3]
    logger.info("  Top 3 documents:")
    for doc_id, score in top_3_docs:
        logger.info(f"    {doc_id}: {score:.4f}")

# ============================================================================
# Step 3: Optional Reranking
# ============================================================================
if not args.no_rerank:
    logger.info(f"\n[Step 3/3] Reranking top-{args.rerank_top_k} documents with CrossEncoder...")
    
    reranker = CrossEncoderReranker(
        model=CrossEncoder(
            "BAAI/bge-reranker-v2-m3",
            trust_remote_code=True,
            # num_labels=1,
            max_length=8192,)
    )
    
    # reranker = CrossEncoderReranker(model=reranker_model)
    
    reranked_results = task.rerank(
        reranker=reranker,
        results=retrieval_results,
        top_k=args.rerank_top_k,
        batch_size=args.batch_size
    )
    
    logger.info(f"✓ Reranking complete for {len(reranked_results)} queries")
    final_results = reranked_results
else:
    logger.info("\n[Step 3/3] Skipping reranking (--no_rerank flag set)")
    final_results = retrieval_results

# ============================================================================
# Step 4: Save Results
# ============================================================================
logger.info("\n[Step 4/4] Saving results...")
task.save_results(top_k=10, output_dir=args.output_dir)

output_path = f"{args.output_dir}/{args.task}/results.csv"
logger.info(f"✓ Results saved to: {output_path}")

# ============================================================================
# Summary
# ============================================================================
logger.info("\n" + "=" * 80)
logger.info("🎉 PIPELINE COMPLETE!")
logger.info("=" * 80)
logger.info(f"Task: {args.task}")
logger.info(f"Method: BM25 Only")
logger.info(f"Retrieved: {args.retrieval_top_k} documents per query")
if not args.no_rerank:
    logger.info(f"Reranked: {args.rerank_top_k} documents per query")
logger.info(f"Results: {output_path}")
logger.info("=" * 80)

print(f"\n✅ Done! Check results in: {output_path}")
