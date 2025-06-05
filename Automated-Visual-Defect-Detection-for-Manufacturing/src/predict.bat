@echo off
set PYTHONPATH=%PYTHONPATH%;.
python predict.py --model_path "models\best_model.pth" %* 