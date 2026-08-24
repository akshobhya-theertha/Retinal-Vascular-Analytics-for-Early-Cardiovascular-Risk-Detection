@echo off
cd /d "%~dp0"
echo Preparing dataset from folder...
python prepare_data.py dataset
if errorlevel 1 echo Prepare step had issues. Ensure dataset folder has class subfolders 0-4 or No_DR, Mild, etc.
echo.
echo Starting training (15 + 20 epochs)...
python train.py --epochs_phase1 15 --epochs_phase2 20 --batch_size 32
pause
