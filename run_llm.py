import sys
from dataclasses import dataclass, field
from typing import List
from typing import List

sys.path.append('distillation')

from arguments import Arguments
from teacher_llm import Teacher, TeacherQwen, TeacherMistral7B, TeacherOutput
from student import LLMModel, StudentCausalModel, StudentOutput
from data_utils import LLMDataset, LLMDataCollator
from loss import (mse_dim_weight_loss, mse_token_weight_loss, cosine_token_weight_loss,
                  mse_token_dim_weight_loss, derivative_loss, orthogonality_loss, cosine_loss)

from llm_train import load_tokenizer
from typing import Optional, Dict, Any
from transformers import AutoTokenizer, AutoModel, AutoConfig, HfArgumentParser
from torch import nn
import torch.nn.functional as F
import torch
import os
import numpy as np
import random


seed=42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

@dataclass
class RuntimeArguments:
    teacher_model: str = 'VoCuc/Qwen2.5-7B-Instruct-Dolly-SFT'
    teacher_tokenizer: str = 'Qwen/Qwen2.5-7B-Instruct'
    student_model: str = 'facebook/opt-2.7b'
    student_tokenizer: str = 'facebook/opt-2.7b'
    teach_device: str = 'cuda:0'
    student_device: str = 'cuda:1'
    teacher_layers_mapping: List[int] = field(default_factory=lambda: [24, 26, 28])
    student_encoder_layers_finetuned: List[int] = field(default_factory=lambda: [28, 30, 32])
    n_encoder_finetuned: int = 32
    teacher_embedding_dimension: int = 3584
    hidden_loss_weights: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    output_dir: str = 'facebook-checkpoint'
    student_model_type: str = 'opt'
    teacher_model_type: str = 'qwen'


runtime_args = HfArgumentParser(RuntimeArguments).parse_args_into_dataclasses()[0]

teacher_layers_mapping = runtime_args.teacher_layers_mapping or [24, 26, 28]
student_encoder_layers_finetuned = runtime_args.student_encoder_layers_finetuned or [28, 30, 32]
hidden_loss_weights = runtime_args.hidden_loss_weights or [1.0, 1.0, 1.0]

args = Arguments(
    train_data='data/llm/dolly/train.jsonl', 
    val_data='data/llm/dolly/dev.jsonl', 
    test_data='data/llm/dolly/valid.jsonl',
    num_labels=1,
    batch_size=2,
    val_batch_size=64,
    
    max_len=256,

    pad_to_multiple_of=4,
    
    knowledge_distillation=True,
    finetune_hidden_states=True,
    output_attentions=True,
   
    teach_device=runtime_args.teach_device,
    student_device=runtime_args.student_device,
    num_train_epochs=5,
    learning_rate=1e-4,
    weight_decay=0.01,
    warmup_ratio=0.1,

    
    orthogonal_loss_weight=0.1,
    hard_label_loss_weight=0.5,
    
    # vector_embedding_warmup_ratio=0.5,

    teacher_layers_mapping=teacher_layers_mapping,
    student_encoder_layers_finetuned=student_encoder_layers_finetuned,
    n_encoder_finetuned=runtime_args.n_encoder_finetuned,
    finetune_embedding=True,
    hidden_loss_weights=hidden_loss_weights,
    teacher_embedding_dimension=runtime_args.teacher_embedding_dimension,

    orthogonal=False,
    span_loss=True,
    der_loss=True,

    span_weight_pooling=True,
    span_loss_weight=True,

    p=1,

    output_dir=runtime_args.output_dir,


    teacher_model=runtime_args.teacher_model,
    teacher_tokenizer=runtime_args.teacher_tokenizer,
    student_model=runtime_args.student_model,
    student_tokenizer=runtime_args.student_tokenizer,
    # student_model='gpt2-sft-checkpoint',
    # student_tokenizer='openai-community/gpt2',

    load_teacher_tokenizer_kwargs={'token': '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'},

    hf_token='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'
)


teacher_sft_path = None
student_sft_path = None

llm_type = ["gpt2", "opt", "llama", "gptj", "llama2", "mistral", "tinyllama", "minicpm", "qwen"]
student_model_type = runtime_args.student_model_type
teacher_model_type = runtime_args.teacher_model_type

load_model_kwargs = {'torch_dtype': torch.bfloat16,
                     'quantization_config': None,
                     'device_map': args.teach_device,
                     'trust_remote_code': True,
                     'output_hidden_states': args.finetune_hidden_states,
                     'output_attentions': args.output_attentions,
                     'attn_implementation': 'sdpa',
                     'token' : args.hf_token}

# teacher_model = TeacherMistral7B(model_name = args.teacher_model, 
#                                         load_model_kwargs = load_model_kwargs,
#                                         export_hidden_state_layers=args.teacher_layers_mapping,
#                                         weight_pooling=args.span_weight_pooling, 
#                                         span_weight=args.span_loss_weight, 
#                                         sft_path=teacher_sft_path)
if runtime_args.teacher_model_type == 'qwen':
    teacher_model = TeacherQwen(model_name = args.teacher_model, 
                                load_model_kwargs = load_model_kwargs,
                                export_hidden_state_layers=args.teacher_layers_mapping, 
                                weight_pooling=args.span_weight_pooling, 
                                span_weight=args.span_loss_weight, 
                                sft_path=teacher_sft_path)
else:
    teacher_model = TeacherMistral7B(model_name = args.teacher_model, 
                                load_model_kwargs = load_model_kwargs,
                                export_hidden_state_layers=args.teacher_layers_mapping, 
                                weight_pooling=args.span_weight_pooling, 
                                span_weight=args.span_loss_weight, 
                                sft_path=teacher_sft_path)

# teacher_model = None

class StudentCausalModelV2(torch.nn.Module):
    def __init__(self, model:LLMModel, model_path, n_encoder_finetuned, 
                 teacher_hidden_size=-1, finetune_embedding=False, orthogonal=True):
        super().__init__()
        self.model = model

        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())

        print('model output_attentions:', model.get_config().output_attentions)
        print('model output_attentions:', model.get_config().output_hidden_states)
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Total parameters: {total_params:,}")
        print(f"Percentage trainable: {100 * trainable_params / total_params:.2f}%")

        self.device = self.model.device

        self.proj_hidden_layers = None

        if teacher_hidden_size > 0:
            proj_list = []
            for i in range(len(self.model.hidden_layer_fineturn)):
                W = nn.Parameter(torch.empty(self.model.model.config.hidden_size, teacher_hidden_size))
                if orthogonal:
                    nn.init.orthogonal_(W)
                else:
                    nn.init.xavier_uniform_(W)
                proj_list.append(W)
            
            self.proj_hidden_layers = nn.ParameterList(proj_list)

            self.proj_embeddings = nn.Parameter(torch.empty(self.model.get_config().hidden_size, teacher_hidden_size))
            if orthogonal:
                nn.init.orthogonal_(self.proj_embeddings)
            else:
                nn.init.xavier_uniform_(self.proj_embeddings)

            hidden_weight_path = os.path.join(model_path, 'proj_hidden_layers.pt')
            if os.path.exists(hidden_weight_path):
                self.proj_hidden_layers = torch.load(hidden_weight_path, weights_only=False)
            
            if os.path.exists(os.path.join(model_path, 'proj_embeddings.pt')):
                self.proj_embeddings = torch.load(os.path.join(model_path, 'proj_embeddings.pt'),
                                                  weights_only=False)

            self.proj_hidden_layers.to(self.device)
            self.proj_embeddings = nn.Parameter(self.proj_embeddings.to(self.device))

    def decode(self, inputs) -> StudentOutput:
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        outputs = self.model(inputs)

        if outputs.hidden_states is not None and self.proj_hidden_layers is not None:
            hidden_states = []
            outputs.embeddings = outputs.hidden_states[-1]
            for i, proj_layer in enumerate(self.proj_hidden_layers):
                hidden_states.append(outputs.hidden_states[i] @ proj_layer)
                
            outputs.hidden_states = hidden_states

        return outputs

    def save(self, path: str):
        self.model.save(path)
        if self.proj_hidden_layers is not None:
            torch.save(self.proj_hidden_layers, os.path.join(path, 'proj_hidden_layers.pt'))

        if self.proj_embeddings is not None:
            torch.save(self.proj_embeddings, os.path.join(path, 'proj_embeddings.pt'))

load_student_model_kwargs = {'device_map': args.student_device,
                          'output_hidden_states': args.finetune_hidden_states,
                          'output_attentions': args.output_attentions,
                          'attn_implementation': 'eager' if args.output_attentions else 'sdpa',
                          }
from types import SimpleNamespace

lora_conf = SimpleNamespace(**{
    "lora_rank": 32,
    "lora_alpha": 64,
    "lora_dropout": 0.1,
    "lora_target_modules": [
        "q_proj",
        "k_proj",
        "v_proj",
    ],
})
llm_model = LLMModel(model_name=args.student_model,
                          load_model_kwargs=load_student_model_kwargs,
                          hidden_layer_fineturn=args.student_encoder_layers_finetuned,
                          weight_pooling=args.span_weight_pooling, 
                          span_weight=args.span_loss_weight, sft_path=student_sft_path, lora_conf=lora_conf)

student_model = StudentCausalModelV2(llm_model, model_path=args.student_model,
                                   n_encoder_finetuned = args.n_encoder_finetuned,
                                   teacher_hidden_size=args.teacher_embedding_dimension,
                                   finetune_embedding=args.finetune_embedding, 
                                   orthogonal=args.orthogonal)

from typing import Type
from torch.utils.data import DataLoader, Dataset
from torch import nn


def get_token_mapping(s_tokenizer, t_tokenizer, device):
    t_vocab = t_tokenizer.get_vocab()
    s_vocab = s_tokenizer.get_vocab()
    t_id_mapping = []
    s_id_mapping = []
    for s_token, s_token_id in s_vocab.items():
        if s_token in t_vocab:
            s_id_mapping.append(s_token_id)
            t_id_mapping.append(t_vocab[s_token])

    return torch.tensor(s_id_mapping, device=device), torch.tensor(t_id_mapping, device=device)


def debug_token_alignment(student_inputs, teacher_inputs, student_tokenizer, teacher_tokenizer,
                          epoch, step, max_segments=3, max_tokens_per_segment=12):
    def _render(name, inputs, tokenizer):
        input_ids = inputs['input_ids'][0]
        attention_mask = inputs['attention_mask'][0]
        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        seq_len = int(attention_mask.sum().item())

        print(f'[{name}] seq_len={seq_len}')
        print(f'[{name}] first tokens: {tokens[:min(len(tokens), 24)]}')

        if 'pooler_safe_idx' not in inputs or 'pooler_mask' not in inputs:
            print(f'[{name}] pooler tensors are missing')
            return

        safe_idx = inputs['pooler_safe_idx'][0]
        pooler_mask = inputs['pooler_mask'][0]
        n_segments = min(max_segments, safe_idx.size(0))

        for seg_idx in range(n_segments):
            segment_mask = pooler_mask[seg_idx].bool()
            segment_positions = safe_idx[seg_idx][segment_mask]
            if segment_positions.numel() == 0:
                continue

            segment_positions = segment_positions.tolist()
            segment_positions = [pos for pos in segment_positions if pos < len(tokens)]
            segment_tokens = [tokens[pos] for pos in segment_positions[:max_tokens_per_segment]]
            segment_text = tokenizer.decode(
                input_ids[segment_positions],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            print(f'[{name}] seg {seg_idx}: idx={segment_positions[:max_tokens_per_segment]}')
            print(f'[{name}] seg {seg_idx}: tok={segment_tokens}')
            print(f'[{name}] seg {seg_idx}: text={repr(segment_text)}')

    print(f'\n=== token align debug: epoch={epoch + 1}, step={step} ===')
    _render('student', student_inputs, student_tokenizer)
    _render('teacher', teacher_inputs, teacher_tokenizer)

    student_segments = student_inputs['pooler_safe_idx'][0]
    teacher_segments = teacher_inputs['pooler_safe_idx'][0]
    student_masks = student_inputs['pooler_mask'][0]
    teacher_masks = teacher_inputs['pooler_mask'][0]
    compare_segments = min(max_segments, student_segments.size(0), teacher_segments.size(0))

    print('--- aligned segment comparison ---')
    for seg_idx in range(compare_segments):
        student_pos = student_segments[seg_idx][student_masks[seg_idx].bool()].tolist()
        teacher_pos = teacher_segments[seg_idx][teacher_masks[seg_idx].bool()].tolist()
        student_pos = [pos for pos in student_pos if pos < student_inputs['input_ids'][0].size(0)]
        teacher_pos = [pos for pos in teacher_pos if pos < teacher_inputs['input_ids'][0].size(0)]

        student_text = student_tokenizer.decode(
            student_inputs['input_ids'][0][student_pos],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ) if student_pos else ''
        teacher_text = teacher_tokenizer.decode(
            teacher_inputs['input_ids'][0][teacher_pos],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ) if teacher_pos else ''

        print(f'[cmp seg {seg_idx}] student={repr(student_text)}')
        print(f'[cmp seg {seg_idx}] teacher={repr(teacher_text)}')
        print(f'[cmp seg {seg_idx}] match={student_text == teacher_text}')

class Trainer:
    def __init__(self, student: StudentCausalModel, 
                 args: Arguments, teacher_model: Teacher = None,
                 hidden_loss_weights = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 8, 10]):
        super().__init__()

        self.student = student.train()
        self.teacher_model = teacher_model

        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        self.mse_loss = nn.MSELoss(reduction='mean')
        
        self.args = args
        self.args.p = max(args.p, 1e-5)

        self.alpha = args.hard_label_loss_weight
        self.temperature = args.temperature

        self.step = 0

        sum_hidden_loss_weights = sum(hidden_loss_weights)
        self.hidden_loss_weights = [w / sum_hidden_loss_weights for w in hidden_loss_weights]

        self.train_loader, self.val_loader, self.test_loader = self.get_data_loader(args)

        self.total_traning_steps = len(self.train_loader) * args.num_train_epochs
        self.embedding_warmup_steps = int(self.total_traning_steps * args.vector_embedding_warmup_ratio)

        self.k = nn.Parameter(torch.tensor(1.0, device=self.student.device)) 

        self.s_vocab_size = self.student.model.model.config.vocab_size
        self.student_loss_function = self.student.model.model.loss_function

        
        self.teacher_lm_head = nn.Linear(self.teacher_model.model.lm_head.in_features,
                                         self.teacher_model.model.lm_head.out_features,
                                         bias=(self.teacher_model.model.lm_head.bias is not None)
                                        ).to(self.student.device)
        self.teacher_lm_head.load_state_dict(self.teacher_model.model.lm_head.state_dict())
        for p in self.teacher_lm_head.parameters():
            p.requires_grad = False

        self.s_id_mapping, self.t_id_mapping = get_token_mapping(self.student_tokenizer, 
                                                                 self.teacher_tokenizer, 
                                                                 device=self.student.device)

    def get_data_loader(self, args: Arguments):
        self.student_tokenizer = load_tokenizer(student_model_type, args.student_tokenizer, 
                                                args.load_student_tokenizer_kwargs)
        self.teacher_tokenizer = load_tokenizer(teacher_model_type, args.teacher_tokenizer, 
                                                args.load_teacher_tokenizer_kwargs)

        train_dataset = LLMDataset(args.train_data, self.student_tokenizer, 
                                   self.teacher_tokenizer, args.max_len // 2)

        train_collate = LLMDataCollator(self.student_tokenizer, self.teacher_tokenizer,
                                       do_train=True, max_len = args.max_len,
                                       pad_to_multiple_of = args.pad_to_multiple_of,
                                       return_tensors = 'pt', padding = True)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, collate_fn=train_collate)

        return train_loader, None, None


    def get_teacher_eval(self, inputs):
        outputs = self.teacher_model.decode(inputs)

        # logits = self.teacher_model.model.lm_head(outputs.hidden_states[-1])

        # outputs.logits = logits.to(self.student.device, non_blocking=True)
  
        if outputs.hidden_states is not None:
            outputs.hidden_states = outputs.hidden_states.to(self.student.device, non_blocking=True)
            
        if outputs.span_weights is not None:
            outputs.span_weights=outputs.span_weights.to(self.student.device, non_blocking=True)

        return outputs

    def soft_label_distill_loss(self, student_logits, teacher_logits, distill_temperature = 2.0):
        
        student_probs = F.log_softmax(student_logits / distill_temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / distill_temperature, dim=-1)

        loss = F.kl_div(student_probs, teacher_probs, reduction='batchmean')
        # loss = loss * (distill_temperature * distill_temperature)

        return loss


    def knowledge_distillation_loss(self, student_outputs: StudentOutput,
                                    teacher_outputs: TeacherOutput = None):
        kd_loss = 0
        temp_loss = torch.tensor(0)

        if teacher_outputs is not None:
            if teacher_outputs.hidden_states is not None:
                span_loss, der_loss = 0, 0
                n_layer = teacher_outputs.hidden_states.size(0)
                span_weights = teacher_outputs.span_weights.squeeze(-1)
                _, B, N = span_weights.size()

                mask = span_weights[-1].bool()  # [B, N]

                span_weights = span_weights ** self.args.p
                span_weights = span_weights / span_weights.sum(-1, keepdim=True)

                pair_weights = span_weights[-1].unsqueeze(2) * span_weights[-1].unsqueeze(1)
                mask = torch.eye(N, device=pair_weights.device).bool()  # (N, N)
                pair_weights[:, mask] = 0.0
                pair_weights = pair_weights / pair_weights.sum(dim=(1, 2), keepdim=True).clamp(min=1e-5)
                
                
                span_weights = span_weights.unsqueeze(-1)
                if self.args.span_loss:
                    for i in range(n_layer):
                        s_hidden = student_outputs.hidden_states[i]
                        t_didden = teacher_outputs.hidden_states[i]
                        span_w = span_weights[i]

                        state_loss = cosine_token_weight_loss(s_hidden, t_didden, span_w)
                        # state_loss = mse_token_weight_loss(s_hidden, t_didden, span_w)
            
                        span_loss += self.hidden_loss_weights[i] * state_loss

                        if torch.isnan(span_loss):
                            print('span_loss nan')
                
                if self.args.der_loss:
                    der_loss = derivative_loss(student_outputs.hidden_states,
                                            teacher_outputs.hidden_states,
                                            teacher_outputs.span_weights) / (n_layer - 1)

                    if torch.isnan(der_loss):
                        print('der_loss nan')

                kd_loss += 10 * (span_loss + der_loss)


                # s_logits = self.student.model.model.lm_head(student_outputs.embeddings)
                # s_hidden = F.normalize(s_logits, dim=-1, eps=1e-5)
                # t_hidden = F.normalize(teacher_outputs.logits, dim=-1, eps=1e-5)
                
                # s_hidden = F.normalize(student_outputs.hidden_states[n_layer - 1], dim=-1, eps=1e-5)
                s_hidden = F.normalize(student_outputs.embeddings, dim=-1, eps=1e-5)
                t_hidden = F.normalize(teacher_outputs.hidden_states[n_layer - 1], dim=-1, eps=1e-5)
                
                student_scores = torch.matmul(s_hidden, s_hidden.transpose(-1, -2))
                teacher_scores = torch.matmul(t_hidden, t_hidden.transpose(-1, -2))
                # score_loss = self.mse_loss(student_scores, teacher_scores)
                score_loss = F.mse_loss(student_scores, teacher_scores, reduction='none')
                # score_loss = (score_loss * pair_mask).sum() / pair_mask.sum()
                score_loss = (score_loss * pair_weights).sum() / B
    
                kd_loss += 50 * score_loss

                # s2t_logits = self.teacher_lm_head(student_outputs.hidden_states[n_layer - 1])
                # t_logits = self.teacher_lm_head(teacher_outputs.hidden_states[n_layer - 1])
                # kd_loss += self.soft_label_distill_loss(s2t_logits, t_logits)

                s_logits = self.student.model.model.lm_head(student_outputs.embeddings)
                t_logits = self.teacher_lm_head(teacher_outputs.hidden_states[n_layer - 1])
                
                s_map_logits = s_logits[:, :, self.s_id_mapping]
                t_map_logits = t_logits[:, :, self.t_id_mapping]
                kd_loss += self.soft_label_distill_loss(s_map_logits, t_map_logits)

                # kd_loss += self.manual_kl_div(s_logits, t_logits)

        return kd_loss, temp_loss.item()

    
    def compute_loss(self, student_inputs, labels, teacher_outputs = None):
        student_outputs = self.student.decode(student_inputs)
        
        hard_loss = self.student_loss_function(student_outputs.logits, 
                                               labels.view(-1), self.s_vocab_size)

        kd_loss, _t_loss_, orthogonal_loss= 0, 0, 0

        if self.args.knowledge_distillation and teacher_outputs is not None:
            kd_loss, _t_loss_ = self.knowledge_distillation_loss(student_outputs, teacher_outputs)

            # if self.args.orthogonal:
            #     if self.args.span_loss or self.args.der_loss:
            #         for W in self.student.proj_hidden_layers:
            #             orthogonal_loss += orthogonality_loss(W)
            #         orthogonal_loss = orthogonal_loss / len(self.student.proj_hidden_layers)

            #     # if self.args.embedding_loss_weight > 0:
            #     #     orthogonal_loss += orthogonality_loss(self.student.proj_embeddings)

            #     kd_loss += self.args.orthogonal_loss_weight * orthogonal_loss

        loss = self.alpha * hard_loss + (1.0 - self.alpha) * kd_loss
        # loss = hard_loss + kd_loss
        # loss = hard_loss

        self.step += 1

        return loss, hard_loss

    
import data_utils

# data_utils.N_SPAN = 128
data_utils.TEACHER_OFFSET = 0
data_utils.STUDENT_OFFSET = 0

trainer = Trainer(student_model, args, teacher_model = teacher_model, 
                  hidden_loss_weights = args.hidden_loss_weights)

from torch import optim
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from transformers import get_scheduler
from concurrent.futures import ThreadPoolExecutor
from itertools import chain

train_loader = trainer.train_loader
val_loader = trainer.val_loader
test_loader = trainer.test_loader

from evaluator import Evaluator


# evaluator = Evaluator(
#     tokenizer_path=args.teacher_tokenizer,
#     model_path=None,
#     sft_lora=None,
#     distilled_lora=None,
#     seeds=[10]
# )

# evaluator.model = trainer.student.model.model

benchmark_configs = {'dolly': 'data/llm/dolly/valid.jsonl',
                     'self_instruct': 'data/llm/self-inst/valid.jsonl',
                     'vicuna': 'data/llm/vicuna/valid.jsonl',
                     'sni': 'data/llm/sinst/11_/valid.jsonl',
                     # 'unni':'data/llm/uinst/11_/valid.jsonl'
                    }
dolly_config = {'dolly': 'data/llm/dolly/valid.jsonl'}

# with torch.cuda.amp.autocast(dtype=torch.float16):
#     results = evaluator.evaluate_multiple_benchmarks(
#         benchmark_configs=benchmark_configs,
#         batch_size=32,
#         max_seq_length=256,
#         max_new_tokens=512
#     )

evaluator = Evaluator(
    tokenizer_path=args.student_tokenizer,
    model_path=None,
    sft_lora=None,
    
    distilled_lora=None,
    device=args.student_device,
    seeds=[10]
)

args.num_train_epochs = 10
GRAD_ACCUM_STEPS = 4

trainer.student.train()
trainer.student.model.train()

# optimizer = optim.AdamW(trainer.student.parameters(), lr=args.learning_rate)

optimizer = optim.AdamW(trainer.student.model.parameters(), lr=args.learning_rate)
optimizer.add_param_group({"params": trainer.student.proj_hidden_layers.parameters(), "lr": 5e-4, "weight_decay": 0.0})
optimizer.add_param_group({"params": [trainer.student.proj_embeddings], "lr": 5e-4, "weight_decay": 0.0})

num_steps = len(train_loader) // GRAD_ACCUM_STEPS + 1
total_traning_steps = num_steps * args.num_train_epochs

scaler = GradScaler()

scheduler = get_scheduler(
    name='cosine_with_min_lr',
    optimizer=optimizer,
    num_warmup_steps=int(total_traning_steps * args.warmup_ratio),
    # num_warmup_steps=0,
    num_training_steps=total_traning_steps,
    scheduler_specific_kwargs={'min_lr': 5e-6}
)

executor = ThreadPoolExecutor(max_workers=1)

best_result = 0

# Training loop
for epoch in range(args.num_train_epochs):
    print(('\n' + '%8s' + '%14s' + '%17s' * 2) % ('epoch', 'memory', 'loss', 'student_loss'))
    p_bar = tqdm(chain(train_loader, [(None, None, None)]), total=len(train_loader) + 1)
    loss_total = 0
    student_loss_total = 0
    step = 0

    teacher_outputs = None
    next_teacher_outputs = None

    student_inputs, teacher_inputs, labels = None, None, None
    next_student_inputs, next_teacher_inputs, next_labels= None, None, None

    for batch in p_bar:
        student_inputs, teacher_inputs, labels = (next_student_inputs, 
                                                  next_teacher_inputs, 
                                                  next_labels)
        teacher_outputs = next_teacher_outputs

        next_student_inputs, next_teacher_inputs, next_labels = batch

        if (args.knowledge_distillation 
            and trainer.teacher_model is not None 
            and next_teacher_inputs is not None):

            next_teacher_inputs['logits_to_keep'] = torch.tensor(1)
            teacher_future = executor.submit(trainer.get_teacher_eval, next_teacher_inputs)
        else:
            teacher_future = None

        if student_inputs is None:
            if args.knowledge_distillation and trainer.teacher_model is not None:
                next_teacher_outputs = teacher_future.result()
            continue

        if step == 0 and teacher_inputs is not None:
            debug_token_alignment(student_inputs,
                                  teacher_inputs,
                                  trainer.student_tokenizer,
                                  trainer.teacher_tokenizer,
                                  epoch,
                                  step)

        # optimizer.zero_grad(set_to_none=True)

        labels = labels.to(trainer.student.device)
        with autocast():
            loss, student_loss = trainer.compute_loss(student_inputs, labels, teacher_outputs)

        scaler.scale(loss / GRAD_ACCUM_STEPS).backward()

        if (step + 1) % GRAD_ACCUM_STEPS == 0:
            # scaler.unscale_(optimizer)
            # torch.nn.utils.clip_grad_norm_(trainer.student.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
    
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        loss_total += loss.item()
        student_loss_total += student_loss.item()
        step += 1

        if teacher_future is not None:
            next_teacher_outputs = teacher_future.result()


        memory = f'{torch.cuda.memory_reserved() / 1E9:.4g}G'  # (GB)
        s = ('%8s' + '%14s' + '%17.5g' * 2) % (f'{epoch + 1}/{args.num_train_epochs}', memory,
                                                loss_total / step, student_loss_total / step)
        p_bar.set_description(s)

        if torch.isnan(loss):
            break

    with torch.cuda.amp.autocast(dtype=torch.float16):
        evaluator.model = trainer.student.model.model
        # result = evaluator.evaluate_benchmark_dataset(
        #     dataset_path='data/llm/dolly/dev.jsonl',
        #     dataset_name='dolly', batch_size=16,
        #     max_seq_length=256, max_new_tokens=512)
        result = evaluator.evaluate_benchmark_dataset(
            dataset_path='data/llm/dolly/valid.jsonl',
            dataset_name='dolly', batch_size=16,
            max_seq_length=256, max_new_tokens=512)
        result = evaluator.evaluate_benchmark_dataset(
            dataset_path='data/llm/vicuna/valid.jsonl',
            dataset_name='vicuna', batch_size=16,
            max_seq_length=256, max_new_tokens=512)

    trainer.student.save(args.output_dir + f'-epoch{epoch}')
        
   

executor.shutdown()

evaluator.model = trainer.student.model.model
result = evaluator.evaluate_benchmark_dataset(
            dataset_path='data/llm/dolly/valid.jsonl',
            dataset_name='vicuna', batch_size=16,
            max_seq_length=256, max_new_tokens=512)

result = evaluator.evaluate_benchmark_dataset(
            dataset_path='data/llm/vicuna/valid.jsonl',
            dataset_name='vicuna', batch_size=16,
            max_seq_length=256, max_new_tokens=512)

result = evaluator.evaluate_benchmark_dataset(
            dataset_path='data/llm/sinst/11_/valid.jsonl',
            dataset_name='SNI', batch_size=16,
            max_seq_length=256, max_new_tokens=512)

result = evaluator.evaluate_benchmark_dataset(
            dataset_path='data/llm/self-inst/valid.jsonl',
            dataset_name='selfinst', batch_size=16,
            max_seq_length=256, max_new_tokens=512)

result = evaluator.evaluate_benchmark_dataset(
            dataset_path='data/llm/dialog/valid.jsonl',
            dataset_name='selfinst', batch_size=8,
            max_seq_length=256, max_new_tokens=512)