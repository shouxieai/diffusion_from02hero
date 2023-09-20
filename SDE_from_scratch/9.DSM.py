import torch

def loss_fn(model, x, marginal_prob_std_f, eps=1e-5):
    random_t = torch.rand(x.shape[0], device=x.device) * (1 - eps) + eps # 10,0000 t
    z = torch.randn_like(x)
    std = marginal_prob_std_f(random_t,)
    perturbed_x = x + z * std[:, None]
    score = model(perturbed_x, random_t) # 10,0000 x and 10,0000 t
    loss  = torch.mean(torch.sum((score * std[:, None]) + z ** 2, dim=1)) # average over sample
    return loss
    