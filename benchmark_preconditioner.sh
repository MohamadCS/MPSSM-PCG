PYTHONPATH="./external/neuralif:$PYTHONPATH" python3 -m src.benchmarks.preconditioner_benchmark \
    --dataset Peptides-struct \
    --max-train 2000 \
    --max-val 500 \
    --max-test 500 \
    --epochs 50 
