"""Prompt templates for CLIP-based MRI real/fake classification."""

import torch


def _to_tensor(output):
    """Extract tensor from CLIP output (handles both raw tensor and model output)."""
    if isinstance(output, torch.Tensor):
        return output
    return output.pooler_output if hasattr(output, "pooler_output") else output[0]


# ── Real MRI prompts ──────────────────────────────────────────────────────────
REAL_PROMPTS = [
    "a real brain MRI scan",
    "a genuine brain MRI image",
    "an authentic medical MRI scan",
    "a clinical MRI of the brain",
    "a diagnostic brain MRI image",
]

# ── Synthetic MRI prompts ─────────────────────────────────────────────────────
SYNTHETIC_PROMPTS = [
    "a synthetic brain MRI image",
    "a computer generated MRI scan",
    "a fake MRI image",
    "an artificially generated brain MRI",
    "a simulated medical MRI image",
]


def build_text_embeddings(model, processor, device="cpu"):
    """Build ensemble-averaged text embeddings for the two classes.

    Returns
    -------
    real_emb : torch.Tensor  — shape (embed_dim,), L2-normalised
    fake_emb : torch.Tensor  — shape (embed_dim,), L2-normalised
    """
    real_emb = _ensemble_embed(model, processor, REAL_PROMPTS, device)
    fake_emb = _ensemble_embed(model, processor, SYNTHETIC_PROMPTS, device)
    return real_emb, fake_emb


def _ensemble_embed(model, processor, prompts, device):
    """Encode a list of prompts and return their mean L2-normalised embedding."""
    inputs = processor(text=prompts, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        text_features = _to_tensor(model.get_text_features(**inputs))  # (N, D)
    # Mean pooling across prompts, then normalise
    mean_emb = text_features.mean(dim=0)
    mean_emb = mean_emb / mean_emb.norm()
    return mean_emb
