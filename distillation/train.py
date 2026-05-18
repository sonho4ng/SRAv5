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


class Trainer:
    def __init__(self, student: StudentBertModel, args: Arguments, 
                 class_dataset_type: Type[Dataset], teacher_model: Teacher = None,
                 hidden_loss_weights = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 8, 10]):
        super().__init__()

        self.student = student.train()
        self.teacher_model = teacher_model

        if args.loss_type == "ce":
            self.criterion = nn.CrossEntropyLoss(reduction='mean')
        elif args.loss_type == "mse":
            self.criterion = nn.MSELoss(reduction='mean')


        self.args = args
        self.args.p = max(args.p, 1e-5)

        self.alpha = args.hard_label_loss_weight

        self.step = 0

        sum_hidden_loss_weights = sum(hidden_loss_weights)
        self.hidden_loss_weights = [w / sum_hidden_loss_weights for w in hidden_loss_weights]

        self.student_tokenizer = AutoTokenizer.from_pretrained(args.student_tokenizer, **args.load_student_tokenizer_kwargs)
        self.teacher_tokenizer = AutoTokenizer.from_pretrained(args.teacher_tokenizer, **args.load_teacher_tokenizer_kwargs)

        self.train_loader, self.val_loader, self.test_loader = self.get_data_loader(args, class_dataset_type)

        self.total_traning_steps = len(self.train_loader) * args.num_train_epochs
        self.embedding_warmup_steps = int(self.total_traning_steps * args.vector_embedding_warmup_ratio)

    def get_data_loader(self, args: Arguments, class_dataset_type: Type[Dataset]):
        train_dataset = class_dataset_type(args.train_data)

        train_collate = DataCollator(self.student_tokenizer, self.teacher_tokenizer,
                                    do_train=True, max_len = args.max_len,
                                    pad_to_multiple_of = args.pad_to_multiple_of,
                                    return_tensors = 'pt', padding = True)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                shuffle=True, collate_fn=train_collate)

        val_dataset = class_dataset_type(args.val_data)

        val_collate = DataCollator(self.student_tokenizer, self.teacher_tokenizer,
                                    do_train=False, max_len = args.max_len,
                                    pad_to_multiple_of = args.pad_to_multiple_of,
                                    return_tensors = 'pt', padding = True)

        val_loader = DataLoader(val_dataset, batch_size=args.val_batch_size, collate_fn=val_collate)

        if args.test_data is not None and len(args.test_data) > 0:
            test_dataset = class_dataset_type(args.test_data)
            test_loader = DataLoader(test_dataset, batch_size=args.val_batch_size, collate_fn=val_collate)
        else:
            test_loader = None

        return train_loader, val_loader, test_loader

    def get_teacher_eval(self, inputs):
        outputs = self.teacher_model.encode(inputs)
        if outputs is None:
            return None

        embeddings = outputs.embeddings
        hidden_states, attentions, span_weights, hidden_dim_weights = None, None, None, None

        if outputs.hidden_states is not None:
            hidden_states = outputs.hidden_states.to(self.student.device, non_blocking=True)
        if outputs.attentions is not None:
            attentions = [tuple(t.to(self.student.device, non_blocking=True) for t in atts)
                          for atts in outputs.attentions]
        if outputs.span_weights is not None:
            span_weights = outputs.span_weights.to(self.student.device, non_blocking=True)
        if outputs.hidden_dim_weights is not None:
            hidden_dim_weights = outputs.hidden_dim_weights.to(self.student.device, non_blocking=True)

        outputs = TeacherOutput(
            embeddings = embeddings.to(self.student.device, non_blocking=True),
            hidden_states = hidden_states,
            attentions = attentions,
            span_weights = span_weights,
            hidden_dim_weights = hidden_dim_weights
        )

        return outputs


    def knowledge_distillation_loss(self, student_outputs: StudentOutput,
                                    teacher_outputs: TeacherOutput = None):
        kd_loss = 0
        temp_loss = torch.tensor(0)
        embeddings = student_outputs.embeddings

        if teacher_outputs is not None:

            if teacher_outputs.hidden_states is not None:
                span_loss, der_loss = 0, 0
                n_layer = teacher_outputs.hidden_states.size(0)
                span_weights = teacher_outputs.span_weights

                span_weights = span_weights.squeeze() ** self.args.p
                span_weights = span_weights / span_weights.sum(-1, keepdim=True)
                span_weights = span_weights.unsqueeze(-1)

                if self.args.span_loss:
                    for i in range(n_layer):
                        s_hidden = student_outputs.hidden_states[i]
                        t_didden = teacher_outputs.hidden_states[i]
                        span_w = span_weights[i]

                        state_loss = cosine_token_weight_loss(s_hidden, t_didden, span_w)
                        span_loss += self.hidden_loss_weights[i] * state_loss


                        if torch.isnan(span_loss):
                            print('span_loss nan')
                
                if self.args.der_loss:
                    der_loss = derivative_loss(student_outputs.hidden_states,
                                            teacher_outputs.hidden_states,
                                            teacher_outputs.span_weights) / (n_layer - 1)

                    if torch.isnan(der_loss):
                        print('der_loss nan')

                kd_loss += span_loss + der_loss


            if self.args.embedding_loss_weight > 0:
                embedding_loss_weight = self.args.embedding_loss_weight

                kd_loss += embedding_loss_weight * cosine_loss(embeddings @ self.student.proj_embeddings,
                                                            teacher_outputs.embeddings)

        return kd_loss, temp_loss.item()


    def compute_loss(self, student_inputs, labels,
                     teacher_outputs: TeacherOutput = None):

        student_outputs = self.student.encode(student_inputs)

        hard_loss = self.criterion(student_outputs.logits, labels)


        kd_loss, _t_loss_, orthogonal_loss= 0, 0, 0

        if self.args.knowledge_distillation and teacher_outputs is not None:
            kd_loss, _t_loss_ = self.knowledge_distillation_loss(student_outputs, teacher_outputs)

            if self.args.orthogonal:
                if self.args.span_loss or self.args.der_loss:
                    for W in self.student.proj_hidden_layers:
                        orthogonal_loss += orthogonality_loss(W)
                    orthogonal_loss = orthogonal_loss / len(self.student.proj_hidden_layers)

                if self.args.embedding_loss_weight > 0:
                    orthogonal_loss += orthogonality_loss(self.student.proj_embeddings)

                kd_loss += self.args.orthogonal_loss_weight * orthogonal_loss

        loss = self.alpha * hard_loss + (1.0 - self.alpha) * kd_loss

        self.step += 1

        return loss, hard_loss
    

def train(args: Arguments, trainer: Trainer, evaluate_function, use_scheduler=True):
    trainer.student.train()
    trainer.student.model.train()

    train_loader = trainer.train_loader
    val_loader = trainer.val_loader
    test_loader = trainer.test_loader


    optimizer = optim.AdamW(trainer.student.parameters(), lr=args.learning_rate)

    num_steps = len(train_loader)
    
    scaler = GradScaler()

    if use_scheduler:
        scheduler = get_scheduler(
            name='cosine_with_min_lr',
            optimizer=optimizer,
            num_warmup_steps=int(trainer.total_traning_steps * args.warmup_ratio),
            num_training_steps=trainer.total_traning_steps,
            scheduler_specific_kwargs={'min_lr': 2e-6}
        )

    best_result = 0

    # Training loop
    for epoch in range(args.num_train_epochs):
        print(('\n' + '%8s' + '%14s' + '%17s' * 2) % ('epoch', 'memory', 'loss', 'student_loss'))
        p_bar = tqdm(train_loader, total=num_steps)
        loss_total = 0
        student_loss_total = 0
        step = 0

        for batch in p_bar:
            student_inputs, teacher_inputs, labels = batch

            teacher_outputs = trainer.get_teacher_eval(teacher_inputs)


            optimizer.zero_grad(set_to_none=True)

            labels = labels.to(trainer.student.device)
            with autocast():
                loss, student_loss = trainer.compute_loss(student_inputs, labels, teacher_outputs)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if use_scheduler:
                scheduler.step()

            loss_total += loss.item()
            student_loss_total += student_loss.item()
            step += 1

            memory = f'{torch.cuda.memory_reserved() / 1E9:.4g}G'  # (GB)

            s = ('%8s' + '%14s' + '%17.5g' * 2) % (f'{epoch + 1}/{args.num_train_epochs}', memory,
                                                    loss_total / step, student_loss_total / step)
            p_bar.set_description(s)

            if torch.isnan(loss):
                break

        with torch.cuda.amp.autocast(dtype=torch.float16):
            eval_results = evaluate_function(trainer.student.model, val_loader)
        print("eval: ", eval_results)
        if test_loader is not None:
            with torch.cuda.amp.autocast(dtype=torch.float16):
                test_results = evaluate_function(trainer.student.model, test_loader)
            print("test: ", test_results)

        # trainer.student.save(args.output_dir + f'-epoch{epoch}')
            
        if 'spearman_corr' in eval_results and eval_results['spearman_corr'] > best_result:
            best_result = eval_results['spearman_corr']
            trainer.student.save(args.output_dir)
            print("✅ Improved (spearman). Saved checkpoint!")
            
        if 'f1_macro' in eval_results and eval_results['f1_macro'] > best_result:
            best_result = eval_results['f1_macro']
            trainer.student.save(args.output_dir)
            print("✅ Improved (F1_macro). Saved checkpoint!")

        
