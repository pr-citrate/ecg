import torch
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def generate_counterfactual(model, x, target_label, lambda_coeff=0.1, steps=100, lr=1e-2, mask=None, device='cpu'):
    """
    Generate a counterfactual example by finding minimal perturbation to x
    such that model predicts target_label=0 (or toggles the given label).

    Args:
        model: nn.Module, hybrid ECG model
        x: torch.Tensor, input signal (1, C, T)
        target_label: int, index of label to flip
        lambda_coeff: float, weight for classification loss
        steps: int, number of optimization steps
        lr: float, learning rate for delta update
        mask: torch.Tensor or None, same shape as x to restrict perturbation region
        device: str or torch.device

    Returns:
        x_cf: torch.Tensor, counterfactual example
        delta: torch.Tensor, optimized perturbation
    """
    model.eval()
    x = x.to(device)
    # Initialize delta
    delta = torch.zeros_like(x, requires_grad=True, device=device)
    optimizer = Adam([delta], lr=lr)

    for step in range(steps):
        optimizer.zero_grad()
        x_cf = x + (delta * mask if mask is not None else delta)
        outputs = model(x_cf)
        logits = outputs['logits'][0, target_label]
        # Want to decrease probability for target_label
        bce = F.binary_cross_entropy_with_logits(logits.unsqueeze(0), torch.tensor([0.], device=device))
        loss = lambda_coeff * torch.norm(delta) + bce
        loss.backward()
        optimizer.step()
    x_cf = x + (delta * mask if mask is not None else delta)
    return x_cf.detach(), delta.detach()


def learn_cavs(concept_examples, non_concept_examples, model, device='cpu'):
    """
    Learn Concept Activation Vectors (CAVs) from examples.

    Args:
        concept_examples: list of torch.Tensor, inputs belonging to concept
        non_concept_examples: list of torch.Tensor, inputs not belonging
        model: nn.Module
        device: str or torch.device

    Returns:
        v_c: np.ndarray, concept activation vector
    """
    # Extract concept bottleneck activations
    def get_concept_features(x_batch):
        with torch.no_grad():
            outputs = model(x_batch.to(device))
        return outputs['concept_scores'].cpu().numpy()

    X_pos = np.vstack([get_concept_features(x.unsqueeze(0)) for x in concept_examples])
    X_neg = np.vstack([get_concept_features(x.unsqueeze(0)) for x in non_concept_examples])
    X = np.vstack([X_pos, X_neg])
    y = np.concatenate([np.ones(len(X_pos)), np.zeros(len(X_neg))])
    # scale features for better convergence
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # Linear classifier with more iterations
    clf = LogisticRegression(max_iter=1000, solver='lbfgs').fit(X_scaled, y)
    # CAV is weight vector
    v_c = clf.coef_[0] / scaler.scale_
    return v_c


def compute_tcav_scores(model, cavs, dataloader, label_indices, device='cpu'):
    """
    Compute TCAV scores for each concept and label.

    Args:
        model: nn.Module
        cavs: dict, concept_name -> np.ndarray (vector of length C)
        dataloader: DataLoader yielding (signals, labels)
        label_indices: list of ints, label indices to evaluate
        device: str or torch.device

    Returns:
        tcav_scores: dict of dicts, tcav_scores[concept][label] = score
    """
    model.eval()
    # Convert CAVs to torch tensors
    cav_tensors = {c: torch.tensor(v, dtype=torch.float, device=device) for c, v in cavs.items()}
    # Initialize counters
    counts = {c: {i: 0 for i in label_indices} for c in cavs}
    totals = {i: 0 for i in label_indices}

    for signals, labels in dataloader:
        signals = signals.to(device)
        signals.requires_grad = False
        outputs = model(signals)
        concept_scores = outputs['concept_scores']  # (batch, C)
        for i in label_indices:
            # gradient of logit_i wrt concept_scores
            logits_i = outputs['logits'][:, i]
            grads = torch.autograd.grad(logits_i.sum(), concept_scores, retain_graph=True)[0]  # (batch, C)
            for c, v in cav_tensors.items():
                dot = (grads * v.unsqueeze(0)).sum(dim=1)  # (batch,)
                # count positive directional derivatives
                counts[c][i] += (dot > 0).sum().item()
            totals[i] += signals.size(0)

    tcav_scores = {c: {} for c in cavs}
    for c in cavs:
        for i in label_indices:
            tcav_scores[c][i] = counts[c][i] / totals[i]
    return tcav_scores
