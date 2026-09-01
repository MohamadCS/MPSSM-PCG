### ROCM
- We run all test on AMD 9070XT using ROCM pytorch, so please install your GPU
  specific libraries whenever you see ROCM in the enviroment file.
- Since I downloaded many packages one by one the final requirements.txt is
  reprodcued using `uv pip freeze > requirements.txt`
- Download the env using `uv venv .venv && uv pip install --python
  .venv/bin/python -r requirements.txt`


### Datasets 
- Please download clone the ECHO datasets repositry `https://github.com/Graph-ECHO-Benchmark/ECHO`, and download 
the ECHO-SSSP dataset using `python scripts/download-all.py --task sssp`, and move the processed dataset to ./data/ECHO/sssp/
- LRGB should be automatically downloaded.
- We don't include the datasets becuase of their large size.

### NeuralIF
- NeuralIF is almost the same, the only change is `__call__` to `forward`, and
  we avoid load numml as we don't need it for our tests. 
- To run a module always use the prefix PYTHONPATH="./external/neuralif:$PYTHONPATH" 


### Running the exact paper benchmarks
- ./run_paper_gcn.sh will run all the long-range propagation tests present in paper
- ./run_paper_preconditioner.sh will run all the preconditioning quality tests present in paper


