#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

OUTPUT_DIR="tinyllama-checkpoint"
mkdir -p "$OUTPUT_DIR"

LOG_FILE="$SCRIPT_DIR/$OUTPUT_DIR/train.log"

# Default options (can be overridden by passing the same --option in the command line)
DEFAULT_ARGS=(
  --teacher_model "VoCuc/Mistral7B_Dolly_SFT"
  --teacher_tokenizer "mistralai/Mistral-7B-v0.1"
  --student_model "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
  --student_tokenizer "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
  --teach_device "cuda:6"
  --student_device "cuda:6"
  --teacher_layers_mapping 28 30 32
  --student_encoder_layers_finetuned 18 20 22
  --n_encoder_finetuned 22
  --teacher_embedding_dimension 4096
  --hidden_loss_weights 1 1 1
  --output_dir tinyllama-checkpoint
  --student_model_type tinyllama
  --teacher_model_type mistral
)

exec > >(tee -a "$LOG_FILE") 2>&1

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting training"
echo "Defaults: ${DEFAULT_ARGS[*]}"

# Forward defaults first, then user-supplied args override where applicable
python run_llm.py "${DEFAULT_ARGS[@]}" "$@"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Training finished"