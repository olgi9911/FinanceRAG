# Step 1: Import necessary libraries
# --------------------------------------
# Import required libraries for document retrieval, reranking, and logging setup.
from sentence_transformers import CrossEncoder
import logging

import argparse

from financerag.rerank import CrossEncoderReranker
from financerag.retrieval import DenseRetrieval, SentenceTransformerEncoder
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


# Step 3: Initialize DenseRetriever model
# -------------------------------------
# Initialize the retrieval model using SentenceTransformers. This model will be responsible
# for encoding both the queries and documents into embeddings.
#
# You can replace 'intfloat/e5-large-v2' with any other model supported by SentenceTransformers.
# For example: 'BAAI/bge-large-en-v1.5', 'Linq-AI-Research/Linq-Embed-Mistral', etc.
encoder_model = SentenceTransformerEncoder(
    model_name_or_path='intfloat/e5-large-v2',
    query_prompt='query: ',
    doc_prompt='passage: ',
)

retrieval_model = DenseRetrieval(
    model=encoder_model
)


# Step 4: Perform retrieval
# ---------------------
# Use the model to retrieve relevant documents for given queries.
retrieval_model = DenseRetrieval(
    model=encoder_model
)

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


# Step 5: Initialize CrossEncoder Reranker
# --------------------------------------
# The CrossEncoder model will be used to rerank the retrieved documents based on relevance.
#
# You can replace 'cross-encoder/ms-marco-MiniLM-L-12-v2' with any other model supported by CrossEncoder.
# For example: 'cross-encoder/ms-marco-TinyBERT-L-2', 'cross-encoder/stsb-roberta-large', etc.
reranker = CrossEncoderReranker(
    model=CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')
)


# Step 6: Perform reranking
# -------------------------
# Rerank the top 100 retrieved documents using the CrossEncoder model.
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


# Step 7: Save results
# -------------------
# Save the results to the specified output directory as a CSV file.
output_dir = './results/baseline'
task.save_results(output_dir=output_dir)

# Confirm the results have been saved.
print("Results have been saved to results.csv")