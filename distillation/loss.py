from torch import torch
from torch.nn import functional as F



def mse_dim_weight_loss(student_hidden_states, teacher_hidden_states, weights):
    squared_diff = (student_hidden_states - teacher_hidden_states) ** 2
    weighted_squared_diff = squared_diff * weights
    mask_nonzero = (student_hidden_states.abs().sum(dim=-1) != 0).float()
    weighted_mse_loss = weighted_squared_diff.sum() / mask_nonzero.sum() / student_hidden_states.shape[-1]

    return weighted_mse_loss

def orthogonality_loss(W):
    m = W.shape[0]
    I = torch.eye(m, device=W.device, dtype=W.dtype)
    diff = W @ W.T - I
    return torch.sum(diff**2)

def derivative_loss(student_hidden_states, teacher_hidden_states, weights):
    loss = 0
    mask_fill = None

    for i in range(len(student_hidden_states) - 1):
        delta_hidden_student = student_hidden_states[i + 1] - student_hidden_states[i]
        delta_hidden_teacher = teacher_hidden_states[i + 1] - teacher_hidden_states[i]

        cos_sim = F.cosine_similarity(delta_hidden_student, delta_hidden_teacher, dim=-1, eps=1e-5)
        cos_sim_loss = 1 - cos_sim
        cos_sim_loss = cos_sim_loss * weights[i + 1].squeeze()

        loss += cos_sim_loss.sum(-1).mean()

    return loss

def mse_token_dim_weight_loss(student_hidden_states, teacher_hidden_states, token_weights, dim_weights):
    squared_diff = (student_hidden_states - teacher_hidden_states) ** 2
    weighted_squared_diff = squared_diff * dim_weights
    weighted_mse_loss = weighted_squared_diff.mean(-1) * token_weights.squeeze()
    weighted_mse_loss = weighted_mse_loss.sum(-1).mean()

    return weighted_mse_loss

def mse_token_weight_loss(student_hidden_states, teacher_hidden_states, token_weights):
    squared_diff = (student_hidden_states - teacher_hidden_states) ** 2
    weighted_mse_loss = squared_diff.mean(-1) * token_weights.squeeze()
    weighted_mse_loss = weighted_mse_loss.sum(-1).mean()

    return weighted_mse_loss

def cosine_token_weight_loss(student_hidden_states, teacher_hidden_states, token_weights):
    cos_sim = F.cosine_similarity(student_hidden_states, teacher_hidden_states, dim=-1, eps=1e-5)
    cos_sim_loss = 1 - cos_sim
    weighted_cos_sim_loss = cos_sim_loss * token_weights.squeeze()
    weighted_mse_loss = weighted_cos_sim_loss.sum(-1).mean()

    return weighted_mse_loss

def cosine_loss(student_embeddings, teacher_embeddings):
    cos_sim = F.cosine_similarity(student_embeddings, teacher_embeddings, dim=-1)
    cos_sim_loss = 1 - cos_sim
    return cos_sim_loss.mean()


        