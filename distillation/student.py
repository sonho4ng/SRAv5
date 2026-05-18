from typing import Optional, Dict, Any
import torch
from transformers import AutoModel, AutoModelForSequenceClassification, AutoModelForCausalLM, AutoConfig
from transformers.modeling_outputs import ModelOutput
from dataclasses import dataclass
from torch import nn, Tensor
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from utils import get_span_hidden_states, get_span_hidden_states_custom
    
import os

import logging
logger = logging.getLogger(__name__)



@dataclass
class StudentOutput(ModelOutput):
    logits: Optional[Tensor] = None
    embeddings: Optional[Tensor] = None
    hidden_states: Any = None
    span_weights: Any = None


class TaskBertModel(torch.nn.Module):
    def __init__(self, hidden_layer_fineturn, weight_pooling=True, span_weight=True):
        super().__init__()

        self.hidden_layer_fineturn = hidden_layer_fineturn
        self.weight_pooling = weight_pooling
        self.span_weight = span_weight

        if weight_pooling and span_weight:
            self.get_span_hidden_states = get_span_hidden_states
        else:
            self.get_span_hidden_states = get_span_hidden_states_custom

    def forward(self, inputs: Dict[str, Tensor] = None):
        raise NotImplementedError
    
    def get_bert_model(self):
        raise NotImplementedError
    
    def get_config(self):
        raise NotImplementedError
    
    def save(self, output_dir: str):
        raise NotImplementedError


class ClassificationBertModel(TaskBertModel):
    def __init__(self, model_name, load_model_kwargs = {}, 
                 hidden_layer_fineturn=[23], weight_pooling=True, span_weight=True):
        super().__init__(hidden_layer_fineturn, weight_pooling, span_weight)

        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, **load_model_kwargs)

        self.device = self.model.device
        

    def forward(self, inputs: Dict[str, Tensor] = None):
        safe_idx = inputs.pop('pooler_safe_idx', None)
        pooler_mask = inputs.pop('pooler_mask', None)

        outputs = self.model(**inputs, return_dict=True)

        if not self.training:
            return StudentOutput(logits=outputs.logits)
        
        if outputs.hidden_states is not None:
            hidden_states = outputs.hidden_states
            embeddings = outputs.hidden_states[-1][:, 0]
        else:
            hidden_states = None
            embeddings = None
        
        attentions =  outputs.attentions
        if attentions is None:
            attentions = torch.ones((self.model.config.num_hidden_layers,
                                     inputs['input_ids'].size(0),
                                     self.model.config.num_attention_heads, 
                                     inputs['input_ids'].size(1),
                                     inputs['input_ids'].size(1)), 
                                     device=inputs['input_ids'].device)

        span_weights = None
        if safe_idx is not None and hidden_states is not None:
            hidden_states, span_weights = self.get_span_hidden_states(inputs, hidden_states, 
                                                                      attentions, safe_idx, 
                                                                      pooler_mask, inputs['attention_mask'],
                                                                      self.hidden_layer_fineturn,
                                                                      self.weight_pooling, self.span_weight)


        return StudentOutput(
            logits=outputs.logits,
            embeddings=embeddings,
            hidden_states=hidden_states,
            span_weights=span_weights
        )

    def save(self, output_dir: str):
        self.model.save_pretrained(output_dir, state_dict=self.model.state_dict())

    def get_bert_model(self):
        return self.model.bert
    
    def get_config(self):
        return self.model.config


class STSBertModel(TaskBertModel):
    def __init__(self, model_name, load_model_kwargs = {}, hidden_layer_fineturn=[23], 
                 weight_pooling=True, span_weight=True):
        super().__init__(hidden_layer_fineturn, weight_pooling, span_weight)

        self.model = AutoModel.from_pretrained(model_name, **load_model_kwargs)

        self.device = self.model.device

        self.hidden_size = self.model.config.hidden_size

        # Create a regression head for STS score prediction (0-5 scale typically)
        self.regressor = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.Tanh(),
            nn.Linear(self.hidden_size // 2, 1)
        )

        regressor_path = os.path.join(model_name, "regressor.pt")
        if os.path.exists(regressor_path):
            regressor_state_dict = torch.load(regressor_path, map_location="cpu")
            self.regressor.load_state_dict(regressor_state_dict)

        self.regressor = self.regressor.to(self.device)

    def forward(self, inputs: Dict[str, Tensor] = None):
        safe_idx = inputs.pop('pooler_safe_idx', None)
        pooler_mask = inputs.pop('pooler_mask', None)

        outputs = self.model(**inputs, return_dict=True)

        # Get the CLS token representation (for sentence embedding)
        pooled_output = outputs.last_hidden_state[:, 0]

        # Apply the regressor to get similarity score
        score = torch.sigmoid(self.regressor(pooled_output)).squeeze() * 5.0

        if not self.training:
            return StudentOutput(logits=score)
        
        if outputs.hidden_states is not None:
            hidden_states = outputs.hidden_states
            embeddings = outputs.hidden_states[-1][:, 0]
        else:
            hidden_states = None
            embeddings = None
        
        attentions =  outputs.attentions
        if attentions is None:
            attentions = torch.ones((self.model.config.num_hidden_layers,
                                     inputs['input_ids'].size(0),
                                     self.model.config.num_attention_heads, 
                                     inputs['input_ids'].size(1),
                                     inputs['input_ids'].size(1)), device=inputs['input_ids'].device)
        
        span_weights = None
        if safe_idx is not None and hidden_states is not None:
            hidden_states, span_weights = self.get_span_hidden_states(inputs, hidden_states, 
                                                                      attentions, safe_idx, 
                                                                      pooler_mask, inputs['attention_mask'],
                                                                      self.hidden_layer_fineturn,
                                                                      self.weight_pooling, self.span_weight)


        return StudentOutput(
            logits=score,
            embeddings=embeddings,
            hidden_states=hidden_states,
            span_weights=span_weights
        )

    def save(self, output_dir: str):
        self.model.save_pretrained(output_dir, state_dict=self.model.state_dict())
        regressor_path = os.path.join(output_dir, "regressor.pt")
        torch.save(self.regressor.state_dict(), regressor_path)

    def get_bert_model(self):
        return self.model
    
    def get_config(self):
        return self.model.config
    

class STSBertModelV2(STSBertModel):
    def __init__(self, model_name, load_model_kwargs = {}, hidden_layer_fineturn=[23], 
                 weight_pooling=True, span_weight=True):
        TaskBertModel.__init__(self, hidden_layer_fineturn, weight_pooling, span_weight)

        self.model = AutoModel.from_pretrained(model_name, **load_model_kwargs)

        self.device = self.model.device

        self.hidden_size = self.model.config.hidden_size

        # Create a regression head for STS score prediction (0-5 scale typically)
        self.regressor = nn.Linear(self.hidden_size, 1)

        regressor_path = os.path.join(model_name, "regressor.pt")
        if os.path.exists(regressor_path):
            regressor_state_dict = torch.load(regressor_path, map_location="cpu")
            self.regressor.load_state_dict(regressor_state_dict)

        self.regressor = self.regressor.to(self.device)


class BertEmbedding(TaskBertModel):
    def __init__(self, model_name, load_model_kwargs = {}, sentence_mean_pooling=False,
                 hidden_layer_fineturn=[23], weight_pooling=True, span_weight=True):
        super().__init__(hidden_layer_fineturn, weight_pooling, span_weight)

        self.model = AutoModel.from_pretrained(model_name, **load_model_kwargs)

        self.device = self.model.device

        self.sentence_mean_pooling = sentence_mean_pooling
        

    def forward(self, inputs: Dict[str, Tensor] = None):
        safe_idx = inputs.pop('pooler_safe_idx', None)
        pooler_mask = inputs.pop('pooler_mask', None)

        outputs = self.model(**inputs, return_dict=True)

        if not self.training:
            return StudentOutput(logits=None)

        if self.sentence_mean_pooling:
            s = torch.sum(outputs.last_hidden_state * inputs['attention_mask'].unsqueeze(-1).float(), 
                          dim=1)
            d = inputs['attention_mask'].sum(axis=1, keepdim=True).float()
            embeddings = s / d
            
        else:
            embeddings = outputs.last_hidden_state[:, 0, :]
        
        if outputs.hidden_states is not None:
            hidden_states = outputs.hidden_states
        else:
            hidden_states = None
        
        attentions =  outputs.attentions
        if attentions is None:
            attentions = torch.ones((self.model.config.num_hidden_layers,
                                     inputs['input_ids'].size(0),
                                     self.model.config.num_attention_heads, 
                                     inputs['input_ids'].size(1),
                                     inputs['input_ids'].size(1)), 
                                     device=inputs['input_ids'].device)

        span_weights = None
        if safe_idx is not None and hidden_states is not None:
            hidden_states, span_weights = self.get_span_hidden_states(inputs, hidden_states, 
                                                                      attentions, safe_idx, 
                                                                      pooler_mask, inputs['attention_mask'],
                                                                      self.hidden_layer_fineturn,
                                                                      self.weight_pooling, self.span_weight)


        return StudentOutput(
            logits=None,
            embeddings=embeddings,
            hidden_states=hidden_states,
            span_weights=span_weights
        )

    def save(self, output_dir: str):
        self.model.save_pretrained(output_dir, state_dict=self.model.state_dict())

    def get_bert_model(self):
        return self.model
    
    def get_config(self):
        return self.model.config


class StudentBertModel(torch.nn.Module):
    def __init__(self, model:TaskBertModel, model_path, n_encoder_finetuned, 
                 teacher_hidden_size=-1, finetune_embedding=False, orthogonal=True):
        super().__init__()
        self.model = model

        if not finetune_embedding or n_encoder_finetuned < model.get_config().num_hidden_layers:
            for k, v in self.model.get_bert_model().embeddings.named_parameters():
                v.requires_grad = False

        for layer in self.model.get_bert_model().encoder.layer[:-n_encoder_finetuned]:
            for name, param in layer.named_parameters():
                param.requires_grad = False

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
                proj_list.append(W)
            
            self.proj_hidden_layers = nn.ParameterList(proj_list)

            self.proj_embeddings = nn.Parameter(torch.empty(self.model.get_config().hidden_size, teacher_hidden_size))
            if orthogonal:
                nn.init.orthogonal_(self.proj_embeddings)

            hidden_weight_path = os.path.join(model_path, 'proj_hidden_layers.pt')
            if os.path.exists(hidden_weight_path):
                self.proj_hidden_layers = torch.load(hidden_weight_path, weights_only=False)
            
            if os.path.exists(os.path.join(model_path, 'proj_embeddings.pt')):
                self.proj_embeddings = torch.load(os.path.join(model_path, 'proj_embeddings.pt'),
                                                  weights_only=False)

            self.proj_hidden_layers.to(self.device)
            self.proj_embeddings = nn.Parameter(self.proj_embeddings.to(self.device))

    def encode(self, inputs) -> StudentOutput:
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        outputs = self.model(inputs)

        if outputs.hidden_states is not None and self.proj_hidden_layers is not None:
            hidden_states = []
            for i, proj_layer in enumerate(self.proj_hidden_layers):
                hidden_states.append(outputs.hidden_states[i] @ proj_layer)
                
            outputs.hidden_states = hidden_states

            # outputs.hidden_states = torch.stack(hidden_states)
            # outputs.hidden_states = self.proj_hidden_layers[-1](outputs.hidden_states)
            # outputs.hidden_states = self.proj_layer(outputs.hidden_states)

        return outputs

    def save(self, path: str):
        self.model.save(path)
        if self.proj_hidden_layers is not None:
            torch.save(self.proj_hidden_layers, os.path.join(path, 'proj_hidden_layers.pt'))

        if self.proj_embeddings is not None:
            torch.save(self.proj_embeddings, os.path.join(path, 'proj_embeddings.pt'))


class LLMModel(torch.nn.Module):
    def __init__(self, model_name, load_model_kwargs = {}, hidden_layer_fineturn=[23], 
                 weight_pooling=True, span_weight=True, lora_conf=None, sft_path=None):
        super().__init__()

        self.hidden_layer_fineturn = hidden_layer_fineturn
        self.weight_pooling = weight_pooling
        self.span_weight = span_weight
        self.lora_config = lora_conf

        if weight_pooling and span_weight:
            self.get_span_hidden_states = get_span_hidden_states
        else:
            self.get_span_hidden_states = get_span_hidden_states_custom

        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        config.output_hidden_states = load_model_kwargs.pop('output_hidden_states', False)
        config.output_attentions = load_model_kwargs.pop('output_attentions', False)
        load_model_kwargs['config'] = config
        
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_model_kwargs)

        if sft_path is not None:
            print("Loading adapter for student")
            self.model = PeftModel.from_pretrained(self.model, sft_path)
            self.model = self.model.merge_and_unload()

        if lora_conf is not None:
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=lora_conf.lora_rank,
                lora_alpha=lora_conf.lora_alpha,
                lora_dropout=lora_conf.lora_dropout,
                target_modules=lora_conf.lora_target_modules
            )
            self.model = get_peft_model(self.model, lora_config).to(self.model.device)
            self.model.print_trainable_parameters()

        self.device = self.model.device

    def forward(self, inputs: Dict[str, Tensor] = None):
        safe_idx = inputs.pop('pooler_safe_idx', None)
        pooler_mask = inputs.pop('pooler_mask', None)

        outputs = self.model(**inputs, return_dict=True)

        if not self.training:
            return StudentOutput(logits=None)
        
        if outputs.hidden_states is not None:
            hidden_states = outputs.hidden_states
        else:
            hidden_states = None
        
        attentions =  outputs.attentions
        if attentions is None:
            attentions = torch.ones((self.model.config.num_hidden_layers,
                                     inputs['input_ids'].size(0),
                                     self.model.config.num_attention_heads, 
                                     inputs['input_ids'].size(1),
                                     inputs['input_ids'].size(1)), 
                                     device=inputs['input_ids'].device)

        span_weights = None
        if safe_idx is not None and hidden_states is not None:
            hidden_states, span_weights = self.get_span_hidden_states(inputs, hidden_states, 
                                                                      attentions, safe_idx, 
                                                                      pooler_mask, inputs['attention_mask'],
                                                                      self.hidden_layer_fineturn,
                                                                      self.weight_pooling, self.span_weight, 
                                                                      is_causal=True)

        return StudentOutput(
            logits=outputs.logits,
            hidden_states=hidden_states,
            span_weights=span_weights
        )

    def save(self, output_dir: str):
        self.model.save_pretrained(output_dir, state_dict=self.model.state_dict())
    
    def get_config(self):
        return self.model.config
    
class StudentCausalModel(torch.nn.Module):
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
