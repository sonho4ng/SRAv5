import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from scipy.stats import spearmanr


def get_span_hidden_states(inputs, hidden_states, attentions, safe_idx, pooler_mask, attention_mask,
                           hidden_layer_fineturn, weight_pooling=True, span_weight=True, is_causal=False):
    batch_size, seq_length = inputs['input_ids'].size()
    batch_idxs = torch.arange(batch_size, device=inputs['input_ids'].device)[:, None, None]

    # batch_size, max_seg, max_len_all = safe_idx.size()
    # g_safe_idx = safe_idx.view(batch_size, max_seg // 4, -1)
    # g_pooler_mask = pooler_mask.view(batch_size, max_seg // 4, -1)

    mask_2d = attention_mask.unsqueeze(1) * attention_mask.unsqueeze(2)   # (B, N, N)
    mask_4d = mask_2d.unsqueeze(1)

    g_safe_idx = safe_idx
    g_pooler_mask = pooler_mask

    hidden_state_pools = []
    span_weights = []
    for i in hidden_layer_fineturn:
        # weights = attentions[i].sum(dim=(1, 2))
        if is_causal:
            weights = attentions[i-1].sum(dim=1)[:, -1].detach()
        else:
            weights = (attentions[i-1] * mask_4d).sum(dim=(1, 2)).detach()

        weights = weights / weights.sum(-1, keepdim=True)
        weights = weights.unsqueeze(-1)[batch_idxs, g_safe_idx] * g_pooler_mask.unsqueeze(-1)

        gathered = hidden_states[i][batch_idxs, g_safe_idx] * g_pooler_mask.unsqueeze(-1)
        gathered = gathered * weights

        hidden_state_mean = gathered.sum(2) / weights.sum(2).clamp(min=1e-5)
        hidden_state_pools.append(hidden_state_mean)
        span_weights.append(weights.sum(2))

    hidden_states = torch.stack(hidden_state_pools)
    span_weights = torch.stack(span_weights)

    return hidden_states, span_weights

def get_span_hidden_states_custom(inputs, hidden_states, attentions, safe_idx, pooler_mask, attention_mask,
                                  hidden_layer_fineturn, weight_pooling=False, span_weight=False, is_causal=False):
    batch_size, seq_length = inputs['input_ids'].size()
    batch_idxs = torch.arange(batch_size, device=inputs['input_ids'].device)[:, None, None]

    g_safe_idx = safe_idx
    g_pooler_mask = pooler_mask

    mask_2d = attention_mask.unsqueeze(1) * attention_mask.unsqueeze(2)   # (B, N, N)
    mask_4d = mask_2d.unsqueeze(1)

    hidden_state_pools = []
    span_weights = []
    for i in hidden_layer_fineturn:
        # weights = attentions[i].sum(dim=(1, 2))
        if is_causal:
            weights = attentions[i-1].sum(dim=1)[:, -1].detach()
        else:
            weights = (attentions[i-1] * mask_4d).sum(dim=(1, 2)).detach()

        weights = weights / weights.sum(-1, keepdim=True)
        weights = weights.unsqueeze(-1)[batch_idxs, g_safe_idx] * g_pooler_mask.unsqueeze(-1)

        gathered = hidden_states[i][batch_idxs, g_safe_idx] * g_pooler_mask.unsqueeze(-1)
        if weight_pooling:
            gathered = gathered * weights
            hidden_state_mean = gathered.sum(2) / weights.sum(2).clamp(min=1e-5)
        else:
            hidden_state_mean = gathered.sum(2) / g_pooler_mask.sum(2, keepdim=True).clamp(min=1e-5)

        hidden_state_pools.append(hidden_state_mean)

        if span_weight:
            span_weights.append(weights.sum(2))
        else:
            # span_weights.append(g_pooler_mask.sum(2, keepdim=True).clamp(max=1.0))
            span_weights.append(weights.sum(2) ** 1e-5)

    hidden_states = torch.stack(hidden_state_pools)
    span_weights = torch.stack(span_weights)

    return hidden_states, span_weights



def calculate_hidden_dim_weight(teacher_hidden_states, power=1.0):
    # teacher tensor of shape (B, N, d_t)
    (batch_size, sequence_length, hidden_size) = teacher_hidden_states.shape
    std_hidden = teacher_hidden_states.std(dim=(0, 1))
    # of shape (d_t)
    mean_std = std_hidden.mean()
    std_scaled = std_hidden / mean_std
    # of shape (d_t)
    # Scaling, if power=0, should be vanilla MSE loss
    std_scaled = std_scaled ** power
    # of shape (d_t)
    # Broadcasting the new weights to the original shape
    std_scaled_unsqueezed = std_scaled.unsqueeze(0).unsqueeze(0)
    # of shape (1, 1, d_t)

    return std_scaled_unsqueezed

@torch.no_grad()
def evaluate_classification(model, loader):
    model.eval()
    all_preds, all_labels = [], []

    for batch in loader:
        inputs, _, labels = batch
        inputs = {key: value.to(model.device) for key, value in inputs.items()}
        
        inputs.update({"output_attentions": False, "output_hidden_states": False})

        outputs = model(inputs)

        preds = torch.argmax(outputs.logits, dim=-1).detach().cpu().numpy()
        true  = labels.numpy()

        all_preds.extend(preds)
        all_labels.extend(true)

    acc  = accuracy_score(all_labels, all_preds)
    f1m  = f1_score(all_labels, all_preds, average="macro")
    prec = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    rec  = recall_score(all_labels, all_preds, average="macro", zero_division=0)

    model.train()

    return {
        "accuracy": acc,
        "f1_macro": f1m,
        "precision_macro": prec,
        "recall_macro": rec
    }

@torch.no_grad()
def evaluate_sts(model, loader):
    model.eval()
    all_preds, all_labels = [], []

    for batch in loader:
        inputs, _, labels = batch
        inputs = {key: value.to(model.device) for key, value in inputs.items()}
        
        inputs.update({"output_attentions": False, "output_hidden_states": False})

        outputs = model(inputs)

        preds = outputs.logits.squeeze().detach().cpu().numpy()
        true  = labels.numpy()

        all_preds.extend(preds)
        all_labels.extend(true)


    # Tính Spearman
    spearman_corr, p_value = spearmanr(all_labels, all_preds)


    model.train()

    return {
        "spearman_corr": spearman_corr,
        "p_value": p_value
    }
