#!/bin/bash

# Create result directory
mkdir -p result

LOG_FILE="result/experiment_log.txt"

# Initialize log file with timestamp
echo "================================================================" | tee -a "$LOG_FILE"
echo "Experiment Run Started: $(date)" | tee -a "$LOG_FILE"
echo "================================================================" | tee -a "$LOG_FILE"

run_experiment() {
    POP_SIZE=$1
    ITERATIONS=$2
    SEED=$3
    DESC=$4

    echo "" | tee -a "$LOG_FILE"
    echo "----------------------------------------------------------------" | tee -a "$LOG_FILE"
    echo "Experiment: $DESC" | tee -a "$LOG_FILE"
    echo "Parameters: Pop=$POP_SIZE, Iter=$ITERATIONS, Seed=$SEED" | tee -a "$LOG_FILE"
    echo "----------------------------------------------------------------" | tee -a "$LOG_FILE"
    
    python examples/example_g12_ga_vs_ma.py --pop_size $POP_SIZE --iterations $ITERATIONS --seed $SEED | tee -a "$LOG_FILE"
}

# Run 100 experiments with random parameters
for i in {1..100}
do
    # Generate random parameters
    # Population size between 20 and 200
    POP_SIZE=$((20 + RANDOM % 181))
    
    # Iterations between 100 and 1000
    ITERATIONS=$((100 + RANDOM % 901))
    
    # Random seed
    SEED=$RANDOM
    
    run_experiment $POP_SIZE $ITERATIONS $SEED "Random Experiment #$i"
done

echo "" | tee -a "$LOG_FILE"
echo "All experiments completed. Results saved to result/ directory." | tee -a "$LOG_FILE"
