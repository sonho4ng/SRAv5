from arguments import Arguments
from teacher import Teacher, CustomsMistralAttention, TeacherLLM2VecMistral7B, TeacherOutput
from student import ClassificationBertModel, STSBertModel, StudentBertModel, StudentOutput
from data_utils import ClassificationDataset, SentencePairClassificationDataset, STSDataset, DataCollator
from loss import (mse_dim_weight_loss, mse_token_weight_loss, cosine_token_weight_loss,
                  mse_token_dim_weight_loss, derivative_loss, orthogonality_loss, cosine_loss)
from utils import evaluate_classification, evaluate_sts

from transformers import AutoTokenizer, AutoModel, AutoConfig, BitsAndBytesConfig
from torch.utils.data import DataLoader, Dataset
from torch import nn
import torch
import os

from torch import optim
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from transformers import get_scheduler
from itertools import chain
from typing import Type


def load_tokenizer(model_type, path, kwargs):        
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, **kwargs)
    if model_type in ["gpt2", "opt", "llama", "gptj", "llama2", "mistral", "tinyllama", "minicpm"]:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token
    elif model_type == "qwen":
        # tokenizer.pad_token_id = 151646
        tokenizer.eos_token_id = 151643
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token
    else:
        print('tokenizer unknow')
    
    return tokenizer

