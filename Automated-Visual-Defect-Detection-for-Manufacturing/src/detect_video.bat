@echo off
set PYTHONPATH=%PYTHONPATH%;.
python detect_video.py --model_path "models\best_model.pth" %* 