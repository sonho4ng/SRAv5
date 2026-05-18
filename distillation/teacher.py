from typing import Optional, Tuple, Any
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
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


def custom_roberta_attention_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        if self.position_embedding_type != "absolute" or head_mask is not None:
            return super().forward(
                hidden_states,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                past_key_value,
                output_attentions,
            )

        bsz, tgt_len, _ = hidden_states.size()

        query_layer = self.transpose_for_scores(self.query(hidden_states))

        # If this is instantiated as a cross-attention module, the keys and values come from an encoder; the attention
        # mask needs to be such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        current_states = encoder_hidden_states if is_cross_attention else hidden_states
        attention_mask = encoder_attention_mask if is_cross_attention else attention_mask

        # Check `seq_length` of `past_key_value` == `len(current_states)` to support prefix tuning
        if is_cross_attention and past_key_value and past_key_value[0].shape[2] == current_states.shape[1]:
            key_layer, value_layer = past_key_value
        else:
            key_layer = self.transpose_for_scores(self.key(current_states))
            value_layer = self.transpose_for_scores(self.value(current_states))
            if past_key_value is not None and not is_cross_attention:
                key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
                value_layer = torch.cat([past_key_value[1], value_layer], dim=2)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # SDPA with memory-efficient backend is broken in torch==2.1.2 when using non-contiguous inputs and a custom
        # attn_mask, so we need to call `.contiguous()` here. This was fixed in torch==2.2.0.
        # Reference: https://github.com/pytorch/pytorch/issues/112577
        if self.require_contiguous_qkv and query_layer.device.type == "cuda" and attention_mask is not None:
            query_layer = query_layer.contiguous()
            key_layer = key_layer.contiguous()
            value_layer = value_layer.contiguous()

        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
        # The tgt_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create
        # a causal mask in case tgt_len == 1.
        is_causal = (
            True if self.is_decoder and not is_cross_attention and attention_mask is None and tgt_len > 1 else False
        )

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_layer,
            key_layer,
            value_layer,
            attn_mask=attention_mask,
            dropout_p=self.dropout_prob if self.training else 0.0,
            is_causal=is_causal,
        )

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, self.all_head_size)

        outputs = (attn_output,)
        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs

@dataclass
class TeacherOutput(ModelOutput):
    embeddings: Optional[Tensor] = None
    hidden_states: Any = None
    attentions: Any = None
    pooler_mask: Any = None
    pooler_idx: Any = None
    attention_mask: Any = None
    span_weights: Any = None
    hidden_dim_weights: Any = None



class Teacher:
    def __init__(self, model_name, load_model_kwargs,
                 export_hidden_state_layers, sentence_mean_pooling=False, 
                 weight_pooling=True, span_weight=True):

        self.model = AutoModel.from_pretrained(model_name, **load_model_kwargs)
        self.model = self.model.eval()
        self.model.config.use_cache = False

        self.device = self.model.device

        self.export_hidden_state_layers = export_hidden_state_layers
        
        self.sentence_mean_pooling = sentence_mean_pooling

        self.weight_pooling = weight_pooling
        self.span_weight = span_weight

        if weight_pooling and span_weight:
            self.get_span_hidden_states = get_span_hidden_states
        else:
            self.get_span_hidden_states = get_span_hidden_states_custom

    def encode(self, inputs) -> TeacherOutput:
        return None


class TeacherBGEM3(Teacher):
    def __init__(self, model_name, load_model_kwargs,
                 export_hidden_state_layers=[2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35], 
                 sentence_mean_pooling=False, weight_pooling=True, span_weight=True):

        print('TeacherBGEM3 loading model ...')
        super().__init__(model_name, load_model_kwargs,
                         export_hidden_state_layers, sentence_mean_pooling, 
                         weight_pooling, span_weight)

    def encode(self, inputs) -> TeacherOutput:
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        safe_idx = inputs.pop('pooler_safe_idx', None)
        pooler_mask = inputs.pop('pooler_mask', None)

        with torch.no_grad():
          outputs = self.model(**inputs)

        if self.sentence_mean_pooling:
            s = torch.sum(outputs.last_hidden_state * inputs['attention_mask'].unsqueeze(-1).float(), 
                          dim=1)
            d = inputs['attention_mask'].sum(axis=1, keepdim=True).float()
            embeddings = s / d
            
        else:
            embeddings = outputs.last_hidden_state[:, 0]
            
        # embeddings = F.normalize(embeddings, p=2, dim=1)

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
                                                                      self.weight_pooling, self.span_weight)
            

        # if attentions is not None:                    
        #     attentions = [attentions[layer_idx] for layer_idx in self.teach_layer_attention]
        attentions = None

        return TeacherOutput(
            embeddings = embeddings,
            hidden_states = hidden_states,
            attentions = attentions,
            pooler_mask = pooler_mask,
            pooler_idx = safe_idx,
            attention_mask = inputs['attention_mask'],
            span_weights = span_weights,
            hidden_dim_weights = torch.stack(hidden_dim_weights) if len(hidden_dim_weights) > 0 else None
        )

class TeacherLLM2VecMistral7B(Teacher):
    def __init__(self, model_name, load_model_kwargs,
                 export_hidden_state_layers=[2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35], 
                 sentence_mean_pooling=False, weight_pooling=True, span_weight=True, sft_path=None):

        print('TeacherLLM2VecMistral7B loading model ...')

        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        config.output_hidden_states = load_model_kwargs.pop('output_hidden_states', False)
        config.output_attentions = load_model_kwargs.pop('output_attentions', False)

        load_model_kwargs['config'] = config

        super().__init__(model_name, load_model_kwargs,
                         export_hidden_state_layers, sentence_mean_pooling, 
                         weight_pooling, span_weight)
        
        self.model = PeftModel.from_pretrained(self.model, "McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp")
        self.model = self.model.merge_and_unload() 

        self.model = PeftModel.from_pretrained(self.model, "McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp-supervised")
        self.model = self.model.merge_and_unload() 

        if sft_path is not None:
            self.model = PeftModel.from_pretrained(self.model, sft_path)
            self.model = self.model.merge_and_unload()

        for i, layer in enumerate(self.model.layers):
                if (i + 1) in self.export_hidden_state_layers: 
                    continue
                layer.self_attn = CustomsMistralAttention(layer.self_attn)

        self.model = self.model.eval()

    def encode(self, inputs):
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        safe_idx = inputs.pop('pooler_safe_idx', None)
        pooler_mask = inputs.pop('pooler_mask', None)

        with torch.no_grad():
          outputs = self.model(**inputs)

        if self.sentence_mean_pooling:
            s = torch.sum(outputs.last_hidden_state * inputs['attention_mask'].unsqueeze(-1).float(), 
                          dim=1)
            d = inputs['attention_mask'].sum(axis=1, keepdim=True).float()
            embeddings = s / d
            
        else:
            embeddings = outputs.last_hidden_state[:, -1]
            
        # embeddings = F.normalize(embeddings, p=2, dim=1)

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
                                                                      self.weight_pooling, self.span_weight)
            
        # if attentions is not None:                    
        #     attentions = [attentions[layer_idx] for layer_idx in self.teach_layer_attention]
        attentions = None

        return TeacherOutput(
            embeddings = embeddings,
            hidden_states = hidden_states,
            attentions = attentions,
            pooler_mask = pooler_mask,
            pooler_idx = safe_idx,
            attention_mask = inputs['attention_mask'],
            span_weights = span_weights,
            hidden_dim_weights = torch.stack(hidden_dim_weights) if len(hidden_dim_weights) > 0 else None
        )

class TeacherQwen3(Teacher):
    def __init__(self, model_name, load_model_kwargs,
                 export_hidden_state_layers=[2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35], 
                 sentence_mean_pooling=False, weight_pooling=True, span_weight=True):

        print('TeacherBGEM3 loading model ...')
        super().__init__(model_name, load_model_kwargs,
                         export_hidden_state_layers, sentence_mean_pooling, 
                         weight_pooling, span_weight)
        
        for i, layer in enumerate(self.model.layers):
                if (i + 1) in self.export_hidden_state_layers: 
                    continue
                layer.self_attn = CustomsQwen3Attention(layer.self_attn)
        
    def last_token_pool(self, last_hidden_states, attention_mask):
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


    def encode(self, inputs) -> TeacherOutput:
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        safe_idx = inputs.pop('pooler_safe_idx', None)
        pooler_mask = inputs.pop('pooler_mask', None)

        with torch.no_grad():
          outputs = self.model(**inputs)

        if self.sentence_mean_pooling:
            s = torch.sum(outputs.last_hidden_state * inputs['attention_mask'].unsqueeze(-1).float(), 
                          dim=1)
            d = inputs['attention_mask'].sum(axis=1, keepdim=True).float()
            embeddings = s / d
            
        else:
            embeddings = self.last_token_pool(outputs.last_hidden_state, inputs['attention_mask'])
            
        # embeddings = F.normalize(embeddings, p=2, dim=1)

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
            embeddings = embeddings,
            hidden_states = hidden_states,
            attentions = attentions,
            pooler_mask = pooler_mask,
            pooler_idx = safe_idx,
            attention_mask = inputs['attention_mask'],
            span_weights = span_weights,
            hidden_dim_weights = torch.stack(hidden_dim_weights) if len(hidden_dim_weights) > 0 else None
        )
