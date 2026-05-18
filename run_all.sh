#!/bin/bash
echo "Installing requirements..."
pip install uv
uv sync
source .venv/bin/activate

echo 'Start run training scripts, logs are being saved to ${OUTPUT_DIR}/train.log'


bash ./opt.sh &
bash ./tinyllama.sh &

wait


echo "=== All done ==="