from typing import Optional, Tuple, Any
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig, AutoModelForCausalLM
from transformers.modeling_outputs import ModelOutput
from dataclasses import dataclass
from torch import Tensor
from utils import get_span_hidden_states, calculate_hidden_dim_weight, get_span_hidden_states_custom
from peft import PeftModel

import logging
logger = logging.getLogger(__name__)



class CustomsQwen3Attention(torch.nn.Module):
    def __init__(self, original_self_attn):
        super().__init__()
        self.original = original_self_attn

    def forward(self, **kwargs):
        kwargs['output_attentions'] = False
        return self.original(**kwargs)

class CustomsMistralAttention(torch.nn.Module):
    def __init__(self, original_self_attn):
        super().__init__()
        self.original = original_self_attn

    def forward(self, **kwargs):
        kwargs['output_attentions'] = False
        return self.original(**kwargs)
    
class CustomsGPT2Attention(torch.nn.Module):
    def __init__(self, original_self_attn):
        super().__init__()
        self.original = original_self_attn

    def forward(self, hidden_states, **kwargs):
        kwargs['output_attentions'] = False
        return self.original(hidden_states, **kwargs)


@dataclass
class TeacherOutput(ModelOutput):
    logits: Optional[Tensor] = None
    hidden_states: Any = None
    attentions: Any = None
    pooler_mask: Any = None
    pooler_idx: Any = None
    attention_mask: Any = None
    span_weights: Any = None
    hidden_dim_weights: Any = None



class Teacher:
    def __init__(self, model_name, load_model_kwargs, export_hidden_state_layers,
                 weight_pooling=True, span_weight=True, sft_path=None):

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_model_kwargs)
        if sft_path is not None:
            self.model = PeftModel.from_pretrained(self.model, sft_path)
            self.model = self.model.merge_and_unload()
        self.model = self.model.eval()
        # self.model.config.use_cache = False

        self.device = self.model.device

        self.export_hidden_state_layers = export_hidden_state_layers
    

        self.weight_pooling = weight_pooling
        self.span_weight = span_weight

        if weight_pooling and span_weight:
            self.get_span_hidden_states = get_span_hidden_states
        else:
            self.get_span_hidden_states = get_span_hidden_states_custom

    def decode(self, inputs) -> TeacherOutput:
        return None


class TeacherMistral7B(Teacher):
    def __init__(self, model_name, load_model_kwargs,
                 export_hidden_state_layers=[2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35], 
                 sentence_mean_pooling=False, weight_pooling=True, span_weight=True, sft_path=None):

        print('TeacherMistral7B loading model ...')

        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        config.output_hidden_states = load_model_kwargs.pop('output_hidden_states', False)
        config.output_attentions = load_model_kwargs.pop('output_attentions', False)
        load_model_kwargs['config'] = config

        super().__init__(model_name, load_model_kwargs,
                         export_hidden_state_layers, 
                         weight_pooling, span_weight, sft_path)

        for i, layer in enumerate(self.model.model.layers):
                if (i + 1) in self.export_hidden_state_layers: 
                    continue
                layer.self_attn = CustomsMistralAttention(layer.self_attn)

        self.model = self.model.eval()

    def decode(self, inputs):
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        safe_idx = inputs.pop('pooler_safe_idx', None)
        pooler_mask = inputs.pop('pooler_mask', None)

        with torch.no_grad():
          outputs = self.model(**inputs)

        hidden_states = outputs.hidden_states
        attentions = outputs.attentions
        if attentions is None:
            attentions = torch.ones((self.model.config.num_hidden_layers,
                                     inputs['input_ids'].size(0),
                                     self.model.config.num_attention_heads, 
                                     inputs['input_ids'].size(1),
                                     inputs['input_ids'].size(1)), device=inputs['input_ids'].device)
            
        hidden_dim_weights = []
        span_weights = None
        if safe_idx is not None and hidden_states is not None:
            for i in self.export_hidden_state_layers:
                dim_weights = calculate_hidden_dim_weight(hidden_states[i])
                hidden_dim_weights.append(dim_weights)

            hidden_states, span_weights = self.get_span_hidden_states(inputs, hidden_states, 
                                                                      attentions, safe_idx, pooler_mask,
                                                                      inputs['attention_mask'], 
                                                                      self.export_hidden_state_layers, 
                                                                      self.weight_pooling, self.span_weight, 
                                                                      is_causal=True)
            
        # if attentions is not None:                    
        #     attentions = [attentions[layer_idx] for layer_idx in self.teach_layer_attention]
        attentions = None

        return TeacherOutput(
            logits = outputs.logits,
            hidden_states = hidden_states,
            attentions = attentions,
            pooler_mask = pooler_mask,
            pooler_idx = safe_idx,
            attention_mask = inputs['attention_mask'],
            span_weights = span_weights,
            hidden_dim_weights = torch.stack(hidden_dim_weights)
        )


class TeacherQwen(Teacher):
    def __init__(self, model_name, load_model_kwargs,
                 export_hidden_state_layers=[2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35],
                   weight_pooling=True, span_weight=True, sft_path=None):

        print('TeacherQwen loading model ...')

        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        config.output_hidden_states = load_model_kwargs.pop('output_hidden_states', False)
        config.output_attentions = load_model_kwargs.pop('output_attentions', False)
        load_model_kwargs['config'] = config

        super().__init__(model_name, load_model_kwargs,
                         export_hidden_state_layers, 
                         weight_pooling, span_weight, sft_path)
        
        for i, layer in enumerate(self.model.model.layers):
            if (i + 1) in self.export_hidden_state_layers: 
                continue
            layer.self_attn = CustomsQwen3Attention(layer.self_attn)


    def decode(self, inputs) -> TeacherOutput:
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        safe_idx = inputs.pop('pooler_safe_idx', None)
        pooler_mask = inputs.pop('pooler_mask', None)

        with torch.no_grad():
          outputs = self.model(**inputs)

        hidden_states = outputs.hidden_states
        attentions = outputs.attentions
        if attentions is None:
            attentions = torch.ones((self.model.config.num_hidden_layers,
                                     inputs['input_ids'].size(0),
                                     self.model.config.num_attention_heads, 
                                     inputs['input_ids'].size(1),
                                     inputs['input_ids'].size(1)), device=inputs['input_ids'].device)

        hidden_dim_weights = []
        span_weights = None
        if safe_idx is not None and hidden_states is not None:
            for i in self.export_hidden_state_layers:
                dim_weights = calculate_hidden_dim_weight(hidden_states[i])
                dim_weights = dim_weights.expand(safe_idx.size(0), safe_idx.size(1), self.model.config.hidden_size)
                hidden_dim_weights.append(dim_weights)

            hidden_states, span_weights = self.get_span_hidden_states(inputs, hidden_states, 
                                                                      attentions, safe_idx, pooler_mask,
                                                                      inputs['attention_mask'], 
                                                                      self.export_hidden_state_layers, 
                                                                      self.weight_pooling, self.span_weight, 
                                                                      is_causal=True)
            

        # if attentions is not None:                    
        #     attentions = [attentions[layer_idx] for layer_idx in self.teach_layer_attention]
        attentions = None

        return TeacherOutput(
            logits = outputs.logits,
            hidden_states = hidden_states,
            attentions = attentions,
            pooler_mask = pooler_mask,
            pooler_idx = safe_idx,
            attention_mask = inputs['attention_mask'],
            span_weights = span_weights,
            hidden_dim_weights = torch.stack(hidden_dim_weights)
        )


class TeacherGPT2(Teacher):
    def __init__(self, model_name, load_model_kwargs,
                 export_hidden_state_layers=[2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35],
                   weight_pooling=True, span_weight=True, sft_path=None):

        print('TeacherQwen loading model ...')

        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        config.output_hidden_states = load_model_kwargs.pop('output_hidden_states', False)
        config.output_attentions = load_model_kwargs.pop('output_attentions', False)
        load_model_kwargs['config'] = config

        super().__init__(model_name, load_model_kwargs,
                         export_hidden_state_layers, 
                         weight_pooling, span_weight, sft_path)
        
        for i, layer in enumerate(self.model.transformer.h):
            if (i + 1) in self.export_hidden_state_layers: 
                continue
            layer.attn = CustomsGPT2Attention(layer.attn)


    def decode(self, inputs) -> TeacherOutput:
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        safe_idx = inputs.pop('pooler_safe_idx', None)
        pooler_mask = inputs.pop('pooler_mask', None)

        with torch.no_grad():
          outputs = self.model(**inputs)

        hidden_states = outputs.hidden_states
        attentions = outputs.attentions
        if attentions is None:
            attentions = torch.ones((self.model.config.num_hidden_layers,
                                     inputs['input_ids'].size(0),
                                     self.model.config.num_attention_heads, 
                                     inputs['input_ids'].size(1),
                                     inputs['input_ids'].size(1)), device=inputs['input_ids'].device)

        hidden_dim_weights = []
        span_weights = None
        if safe_idx is not None and hidden_states is not None:
            for i in self.export_hidden_state_layers:
                dim_weights = calculate_hidden_dim_weight(hidden_states[i])
                dim_weights = dim_weights.expand(safe_idx.size(0), safe_idx.size(1), self.model.config.hidden_size)
                hidden_dim_weights.append(dim_weights)

            hidden_states, span_weights = self.get_span_hidden_states(inputs, hidden_states, 
                                                                      attentions, safe_idx, pooler_mask,
                                                                      inputs['attention_mask'], 
                                                                      self.export_hidden_state_layers, 
                                                                      self.weight_pooling, self.span_weight, 
                                                                      is_causal=True)
            

        # if attentions is not None:                    
        #     attentions = [attentions[layer_idx] for layer_idx in self.teach_layer_attention]
        attentions = None

        return TeacherOutput(
            logits = outputs.logits,
            hidden_states = hidden_states,
            attentions = attentions,
            pooler_mask = pooler_mask,
            pooler_idx = safe_idx,
            attention_mask = inputs['attention_mask'],
            span_weights = span_weights,
            hidden_dim_weights = torch.stack(hidden_dim_weights)
        )

