@echo off
REM Activate virtual environment and start training
call venv\Scripts\activate.bat
python train_flowers.py --model resnet18 --epochs 30
pause

