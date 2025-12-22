"""
Example: Using LangChain's EnsembleRetriever for Hybrid Retrieval

This script demonstrates how to use LangChain's built-in EnsembleRetriever
which combines BM25 and Dense retrieval using Reciprocal Rank Fusion (RRF).

Benefits:
- No need to implement fusion logic yourself
- Uses LangChain's battle-tested BM25 implementation
- Automatic RRF (Reciprocal Rank Fusion)
- Persistent ChromaDB storage for fast subsequent runs

Usage:
    # Basic usage
    python qwen3_langchain_hybrid.py --task FinQA

    # With custom weights
    python qwen3_langchain_hybrid.py --task FinQA --bm25_weight 0.3 --dense_weight 0.7

    # Disable ChromaDB persistence (in-memory only)
    python qwen3_langchain_hybrid.py --task FinQA --no_persist

Installation:
    pip install langchain langchain-community chromadb
"""

import logging
import argparse
from sentence_transformers import CrossEncoder

from financerag.retrieval import SentenceTransformerEncoder
from financerag.retrieval.langchain_hybrid import LangChainHybridRetrieval
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
parser = argparse.ArgumentParser(
    description="Run FinanceRAG with LangChain EnsembleRetriever (BM25 + Dense + RRF)"
)
parser.add_argument("--task", type=str, choices=TASK.keys(), default="FinQA", help="Specify the task to run")
parser.add_argument("--bm25_weight", type=float, default=0.5, help="Weight for BM25 retriever (0-1, will be normalized)")
parser.add_argument("--dense_weight", type=float, default=0.5, help="Weight for dense retriever (0-1, will be normalized)")
parser.add_argument("--batch_size", type=int, default=4, help="Batch size for encoding")
parser.add_argument("--retrieval_top_k", type=int, default=200, help="Number of documents to retrieve before reranking")
parser.add_argument("--rerank_top_k", type=int, default=100, help="Number of documents to rerank")
parser.add_argument("--persist_directory", type=str, default="./chroma", help="Directory for ChromaDB persistence")
parser.add_argument("--no_persist", action="store_true", help="Disable ChromaDB persistence (in-memory only)")
parser.add_argument("--output_dir", type=str, default="./results/hybrid", help="Directory to save results")
parser.add_argument("--reranker", type=str, default="BAAI/bge-reranker-v2-m3", help="Cross-Encoder model for reranking")

args = parser.parse_args()

logger.info("=" * 80)
logger.info("FinanceRAG with LangChain EnsembleRetriever")
logger.info("=" * 80)
logger.info(f"Task: {args.task}")
logger.info(f"BM25 Weight: {args.bm25_weight}")
logger.info(f"Dense Weight: {args.dense_weight}")
logger.info(f"ChromaDB: {'In-memory' if args.no_persist else args.persist_directory}")
logger.info("=" * 80)

# ============================================================================
# Step 1: Initialize Task
# ============================================================================
logger.info("\n[Step 1/4] Initializing task...")
task = TASK[args.task]()
logger.info(f"✓ Loaded {len(task.corpus)} documents and {len(task.queries)} queries")

# ============================================================================
# Step 2: Initialize Encoder
# ============================================================================
logger.info("\n[Step 2/4] Initializing encoder model...")
encoder = SentenceTransformerEncoder(
    model_name_or_path='Qwen/Qwen3-Embedding-4B',
    query_prompt='query',
)
logger.info("✓ Encoder initialized: Qwen/Qwen3-Embedding-4B")

# ============================================================================
# Step 3: Create LangChain Hybrid Retriever and Retrieve
# ============================================================================
logger.info("\n[Step 3/4] Creating LangChain EnsembleRetriever...")

persist_dir = None if args.no_persist else args.persist_directory

try:
    hybrid_retriever = LangChainHybridRetrieval(
        encoder_model=encoder,
        bm25_weight=args.bm25_weight,
        dense_weight=args.dense_weight,
        persist_directory=persist_dir,
        collection_name=f"finance_rag_{args.task.lower()}",
        batch_size=args.batch_size
    )
    
    logger.info("✓ LangChain EnsembleRetriever created")
    logger.info(f"  - BM25 weight: {hybrid_retriever.bm25_weight:.2f}")
    logger.info(f"  - Dense weight: {hybrid_retriever.dense_weight:.2f}")
    logger.info(f"  - Fusion method: Reciprocal Rank Fusion (RRF)")
    
    logger.info(f"\nRetrieving top-{args.retrieval_top_k} documents per query...")
    retrieval_results = task.retrieve(
        retriever=hybrid_retriever,
        top_k=args.retrieval_top_k
    )
    
    # Log statistics
    logger.info("\n--- Retrieval Statistics ---")
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

except ImportError as e:
    logger.error("\n❌ LangChain packages not installed!")
    logger.error("Install with: pip install langchain langchain-community chromadb")
    logger.error(f"Error: {e}")
    exit(1)

# ============================================================================
# Step 4: Rerank with Cross-Encoder
# ============================================================================
logger.info(f"\n[Step 4/4] Reranking top-{args.rerank_top_k} documents...")

reranker_model = CrossEncoder(
    args.reranker,
    trust_remote_code=True,
    # num_labels=1,
    # max_length=1024, # 8192
)

reranker = CrossEncoderReranker(model=reranker_model)

reranked_results = task.rerank(
    reranker=reranker,
    results=retrieval_results,
    top_k=args.rerank_top_k,
    batch_size=args.batch_size
)

logger.info(f"✓ Reranking complete for {len(reranked_results)} queries")

# ============================================================================
# Step 5: Save Results
# ============================================================================
logger.info("\n[Step 5/5] Saving results...")
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
logger.info(f"Method: LangChain EnsembleRetriever (BM25 + Dense + RRF)")
logger.info(f"Weights: BM25={hybrid_retriever.bm25_weight:.2f}, Dense={hybrid_retriever.dense_weight:.2f}")
logger.info(f"Retrieved: {args.retrieval_top_k} documents per query")
logger.info(f"Reranked: {args.rerank_top_k} documents per query")
logger.info(f"Results: {output_path}")
logger.info("=" * 80)

print(f"\n✅ Done! Check results in: {output_path}")
