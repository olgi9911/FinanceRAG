#!/bin/bash

# cd /home/kevin/FinanceRAG && source /home/kevin/miniconda3/etc/profile.d/conda.sh && conda activate financerag_env && python BGE.py --task FinanceBench
# ./run_all_tasks.sh
# python submit.py --method baseline

# Activate conda environment
source /home/kevin/miniconda3/etc/profile.d/conda.sh
conda activate financerag_env

# Change to the script directory
cd /home/kevin/FinanceRAG

# List of tasks to run
tasks=("ConvFinQA" "FinanceBench" "FinDER" "FinQA" "FinQABench" "MultiHiertt" "TATQA")

# Run each task with nohup
for task in "${tasks[@]}"; do
    echo "Starting task: $task"
    nohup python BGE.py --task $task > logs/${task}_$(date +%Y%m%d_%H%M%S).log 2>&1 &
    echo "Task $task started with PID: $!"
    
    # Optional: Add a small delay between starting tasks
    sleep 2
done

echo "All tasks started. Check logs/ directory for output."
echo "Use 'ps aux | grep BGE.py' to see running processes"
echo "Use 'tail -f logs/<task>_*.log' to monitor progress"
