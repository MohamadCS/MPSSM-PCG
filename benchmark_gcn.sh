PYTHONPATH="./external/neuralif:$PYTHONPATH" python3 -m src.benchmarks.gcn_benchmark \
    --epochs 300 \
    --pre-epochs 30 \
    --batch-size 32 \
    --gcn-num-layers 1 \
    --pgcn-num-layers 1 \
    --alpha 8.0 \
    --hidden-dim 384 \
    --dataset ECHO-SSSP
