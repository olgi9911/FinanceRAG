#!/bin/bash
# Run hybrid retrieval (BM25 + Dense + RRF) using LangChain EnsembleRetriever
# ChromaDB persistence enabled for faster subsequent runs

python qwen3_langchain_hybrid.py --task ConvFinQA --batch_size 4 --bm25_weight 0.5 --dense_weight 0.5 --persist_directory ./chroma --reranker "BAAI/bge-reranker-v2-m3"
python qwen3_langchain_hybrid.py --task FinanceBench --batch_size 4 --bm25_weight 0.5 --dense_weight 0.5 --persist_directory ./chroma --reranker "jinaai/jina-reranker-v2-base-multilingual"
python qwen3_langchain_hybrid.py --task FinDER --batch_size 1 --bm25_weight 0.5 --dense_weight 0.5 --persist_directory ./chroma --reranker "BAAI/bge-reranker-v2-m3"
python qwen3_langchain_hybrid.py --task FinQA --batch_size 4 --bm25_weight 0.5 --dense_weight 0.5 --persist_directory ./chroma --reranker "Alibaba-NLP/gte-multilingual-reranker-base"
python qwen3_langchain_hybrid.py --task FinQABench --batch_size 4 --bm25_weight 0.5 --dense_weight 0.5 --persist_directory ./chroma --reranker "jinaai/jina-reranker-v2-base-multilingual"
python qwen3_langchain_hybrid.py --task MultiHiertt --batch_size 2 --bm25_weight 0.5 --dense_weight 0.5 --persist_directory ./chroma --reranker "jinaai/jina-reranker-v2-base-multilingual"
python qwen3_langchain_hybrid.py --task TATQA --batch_size 2 --bm25_weight 0.5 --dense_weight 0.5 --persist_directory ./chroma --reranker "BAAI/bge-reranker-v2-m3"
