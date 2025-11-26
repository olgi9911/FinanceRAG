import os
import pandas as pd
import argparse
from pathlib import Path
import subprocess

def combine_results(method):
    """
    Combine CSV results from each dataset's folder into a single submission.csv
    
    Args:
        method: The method name (e.g., 'baseline', 'advanced')
    """
    results_base = Path(f"results/{method}")
    
    # List of datasets
    datasets = ["ConvFinQA", "FinanceBench", "FinDER", "FinQA", "FinQABench", "MultiHiertt", "TAT-QA"]
    
    all_results = []
    
    for dataset in datasets:
        result_file = results_base / dataset / "results.csv"
        if result_file.exists():
            df = pd.read_csv(result_file)
            all_results.append(df)
            print(f"Loaded {len(df)} rows from {result_file}")
        else:
            print(f"Warning: {result_file} not found")
    
    if not all_results:
        raise ValueError(f"No result files found for method: {method}")
    
    # Combine all results
    combined_df = pd.concat(all_results, ignore_index=True)
    
    # Save submission file
    submission_path = results_base / "submission.csv"
    combined_df.to_csv(submission_path, index=False)
    print(f"\nCombined {len(combined_df)} total rows")
    print(f"Saved submission to: {submission_path}")
    
    return submission_path

def submit_to_kaggle(submission_path, message):
    """
    Submit the results to Kaggle competition
    
    Args:
        submission_path: Path to the submission CSV file
        message: Submission message
    """
    cmd = [
        "kaggle", "competitions", "submit",
        "-c", "icaif-24-finance-rag-challenge",
        "-f", str(submission_path),
        "-m", message
    ]
    
    print(f"\nSubmitting to Kaggle: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    
    return result.returncode == 0

def main():
    parser = argparse.ArgumentParser(description="Combine results and submit to Kaggle")
    parser.add_argument("--method", type=str, required=True, 
                        help="Method name (e.g., baseline, advanced)")
    parser.add_argument("--message", type=str, default="Submission",
                        help="Submission message for Kaggle")
    parser.add_argument("--no-submit", action="store_true",
                        help="Only combine results without submitting")
    
    args = parser.parse_args()
    
    # Combine results
    submission_path = combine_results(args.method)
    
    # Submit to Kaggle
    if not args.no_submit:
        success = submit_to_kaggle(submission_path, args.message)
        if success:
            print("\n✓ Submission successful!")
        else:
            print("\n✗ Submission failed!")
    else:
        print("\nSkipping Kaggle submission (--no-submit flag set)")

if __name__ == "__main__":
    main()