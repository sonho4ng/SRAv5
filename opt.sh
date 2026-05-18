#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

OUTPUT_DIR="facebook-checkpoint"
mkdir -p "$OUTPUT_DIR"

LOG_FILE="$SCRIPT_DIR/$OUTPUT_DIR/train.log"


# Default options (can be overridden by passing the same --option in the command line)
DEFAULT_ARGS=(
  --teacher_model "VoCuc/Qwen2.5-7B-Instruct-Dolly-SFT"
  --teacher_tokenizer "Qwen/Qwen2.5-7B-Instruct"
  --student_model "facebook/opt-2.7b"
  --student_tokenizer "facebook/opt-2.7b"
  --teach_device "cuda:5"
  --student_device "cuda:5"
  --teacher_layers_mapping 26 28
  --student_encoder_layers_finetuned 30 32
  --n_encoder_finetuned 32
  --teacher_embedding_dimension 3584
  --hidden_loss_weights 1 1
  --output_dir facebook-checkpoint
  --student_model_type opt
  --teacher_model_type qwen
)

exec > >(tee -a "$LOG_FILE") 2>&1

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting training"
echo "Defaults: ${DEFAULT_ARGS[*]}"

# Forward defaults first, then user-supplied args override where applicable
python run_llm.py "${DEFAULT_ARGS[@]}" "$@"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Training finished"