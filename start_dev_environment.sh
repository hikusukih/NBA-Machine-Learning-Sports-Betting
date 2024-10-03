#!/bin/bash
# Navigate to the directory where main.py is located if it's not in the same directory as this script
cd "C:\Users\david\code\NBA-Machine-Learning-Sports-Betting"
# Start the Jupyter notebook
jupyter notebook &

cd mlflow
# Start the MLflow server
python -m mlflow server --host 127.0.0.1 --port 8765 &

# Wait a few seconds to ensure MLflow server is up and running
sleep 5

# Open the MLflow server in the default web browser
# Use `xdg-open` for Linux/WSL, `start` for Windows, or `open` for MacOS.
start http://127.0.0.1:8765 &

cd "../src"
# Run your main Python script
python new_main.py &

# Wait for all background processes to complete (optional)
wait
