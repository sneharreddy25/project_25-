@echo off
mkdir models 2>nul
set PYTHONPATH=%PYTHONPATH%;.
python -m src.train --data_dir dataset --model_dir models --epochs 50 --batch_size 16 --lr 0.001 