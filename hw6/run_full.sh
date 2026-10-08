#!/bin/sh
# Full HW6 run (resumes from checkpoints if interrupted). Run from hw6/.
cd "$(dirname "$0")"
echo "START $(date)" >> logs/nbconvert.log
caffeinate -dimsu .venv/bin/jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=hw6 --ExecutePreprocessor.timeout=-1 ssl_stl10.ipynb >> logs/nbconvert.log 2>&1
echo "EXIT $? $(date)" >> logs/nbconvert.log
