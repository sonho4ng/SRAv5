import argparse
import kagglehub
from arguments import Arguments
from teacher import TeacherLLM2VecMistral7B
from student import ClassificationBertModel, STSBertModel, StudentBertModel, STSBertModelV2
from data_utils import ClassificationDataset, SentencePairClassificationDataset, STSDataset

from utils import evaluate_classification, evaluate_sts
from train import Trainer, train

from transformers import AutoTokenizer, AutoModel, HfArgumentParser
from torch import nn
import torch
import os
import numpy as np
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def main():
    hf_parser = HfArgumentParser(Arguments)
    args, remaining = hf_parser.parse_args_into_dataclasses(return_remaining_strings=True)

    args: Arguments = args
    args.knowledge_distillation = True
    args.finetune_hidden_states = True
    args.output_attentions = True
    args.weight_decay = 0.01
    args.warmup_ratio = 0.1
    args.orthogonal_loss_weight = 1000
    args.n_encoder_finetuned = 12
    args.finetune_embedding = True


    extra_parser = argparse.ArgumentParser(add_help=False)
    extra_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    extra_parser.add_argument("--teacher_sft", type=str, default=None)
    extra_parser.add_argument("--task_type", type=str, default=None)
    extra_parser.add_argument("--teacher_mean_pooling", type=bool, default=True)

    extras = extra_parser.parse_args(remaining)

    set_seed(extras.seed)

    if extras.task_type == 'classification':
        BERT_MODEL_TASK = ClassificationBertModel
        evaluate_function = evaluate_classification
        class_dataset = ClassificationDataset
    elif extras.task_type == 'sententce_pair_classification':
        BERT_MODEL_TASK = ClassificationBertModel
        evaluate_function = evaluate_classification
        class_dataset = SentencePairClassificationDataset
    elif extras.task_type == 'sts':
        BERT_MODEL_TASK = STSBertModel
        evaluate_function = evaluate_sts
        class_dataset = STSDataset
    else:
        raise ValueError(f"Task type {extras.task_type} not supported")

    if extras.teacher_sft is not None:
        sft_path = kagglehub.dataset_download(extras.teacher_sft)
    else:
        sft_path = None

    print(args)
    print(extras)
    

    # load_model_kwargs = {'torch_dtype': torch.float16,
    #                     'quantization_config': None,
    #                     'device_map': args.teach_device,
    #                     'trust_remote_code': True,
    #                     'output_hidden_states': args.finetune_hidden_states,
    #                     'output_attentions': args.output_attentions,
    #                     'attn_implementation': 'sdpa',
    #                     'token' : args.hf_token}

    # teacher_model = TeacherLLM2VecMistral7B(model_name = args.teacher_model, 
    #                                         load_model_kwargs = load_model_kwargs,
    #                                         export_hidden_state_layers=args.teacher_layers_mapping, 
    #                                         sentence_mean_pooling=extra_parser.teacher_mean_pooling, 
    #                                         weight_pooling=args.span_weight_pooling, 
    #                                         span_weight=args.span_loss_weight, 
    #                                         sft_path=sft_path)


    # load_bert_model_kwargs =   {'device_map': args.student_device,
    #                             'output_hidden_states': args.finetune_hidden_states,
    #                             'output_attentions': args.output_attentions,
    #                             'attn_implementation': 'eager' if args.output_attentions else 'sdpa'}
    
    # bert_model_task = BERT_MODEL_TASK(model_name=args.student_model,
    #                                 load_model_kwargs=load_bert_model_kwargs,
    #                                 hidden_layer_fineturn=args.student_encoder_layers_finetuned,
    #                                 weight_pooling=args.span_weight_pooling, 
    #                                 span_weight=args.span_loss_weight)

    # student_model = StudentBertModel(bert_model_task, model_path=args.student_model,
    #                                 n_encoder_finetuned = args.n_encoder_finetuned,
    #                                 teacher_hidden_size=args.teacher_embedding_dimension,
    #                                 finetune_embedding=args.finetune_embedding, orthogonal=args.orthogonal)
    
    # trainer = Trainer(student_model, args, class_dataset_type = class_dataset, 
    #               teacher_model = teacher_model, hidden_loss_weights = args.hidden_loss_weights)
    
    # train(args, trainer, evaluate_function)


if __name__ == "__main__":
    main()