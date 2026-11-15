#!/bin/bash
# Build the three pools, then run the schedule (two runs at a time on one GPU).
cd ~/local/mathllm
export CUDA_MPS_PIPE_DIRECTORY=$PWD/mps/pipe CUDA_MPS_LOG_DIRECTORY=$PWD/mps/log TORCHINDUCTOR_CACHE_DIR=$PWD/.inductor_cache SCHED_SCRIPT=train3.py
.venv/bin/python build_data3.py 2000000 hard/full.npz full > logs/build_full.log 2>&1
.venv/bin/python build_data3.py 2000000 hard/none.npz none > logs/build_none.log 2>&1
nohup .venv/bin/python sched.py jobs_hard.txt runs/hard 2 > logs/sched_hard.log 2>&1 &
.venv/bin/python build_data3.py 2000000 hard/values.npz values > logs/build_values.log 2>&1
.venv/bin/python eval_old.py runs/v2/v23_11m_s0.pt hard/full.npz runs/hard/old_11m.hard.json > logs/eval_old.log 2>&1
