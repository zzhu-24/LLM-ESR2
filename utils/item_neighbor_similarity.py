import csv
from pathlib import Path

import torch
import torch.nn.functional as F


TOPK = 20


def get_model_item_embeddings(model, model_name, item_num):
    """Return the item representation used immediately before SASRec."""
    model = model.module if hasattr(model, "module") else model
    if model_name == "sasrec":
        return model.item_emb.weight[1:item_num + 1].detach()
    if model_name == "llm_adapter_sasrec":
        llm_embeddings = model.llm_item_emb.weight[1:item_num + 1]
        return model.adapter(llm_embeddings).detach()
    raise ValueError(f"Unsupported model for neighbor tracking: {model_name}")


def mean_topk_by_group(embeddings, popularity, threshold, batch_size):
    """Mean cosine similarity at each neighbor rank for head and tail items."""
    device = embeddings.device
    popularity = torch.as_tensor(
        popularity[1:embeddings.shape[0] + 1], device=device
    )
    active_mask = popularity > 0
    embeddings = embeddings[active_mask]
    popularity = popularity[active_mask]

    if embeddings.shape[0] <= TOPK:
        raise ValueError(f"Top-{TOPK} requires more than {TOPK} active items.")

    normalized = F.normalize(embeddings, p=2, dim=1)
    group_masks = {
        "Head": popularity >= threshold,
        "Tail": popularity < threshold,
    }
    sums = {
        group: torch.zeros(TOPK, dtype=torch.float64, device=device)
        for group in group_masks
    }
    counts = {group: 0 for group in group_masks}

    with torch.no_grad():
        for start in range(0, normalized.shape[0], batch_size):
            end = min(start + batch_size, normalized.shape[0])
            similarities = normalized[start:end] @ normalized.T
            rows = torch.arange(end - start, device=device)
            similarities[rows, rows + start] = -torch.inf
            top_values = torch.topk(
                similarities, k=TOPK, dim=1, largest=True, sorted=True
            ).values

            for group, mask in group_masks.items():
                batch_mask = mask[start:end]
                if torch.any(batch_mask):
                    sums[group] += top_values[batch_mask].double().sum(dim=0)
                    counts[group] += int(batch_mask.sum().item())

    for group, count in counts.items():
        if count == 0:
            raise ValueError(f"No {group.lower()} items for ts_item={threshold}.")
    means = {group: (sums[group] / counts[group]).cpu() for group in sums}
    return means, counts


def initialize_trace(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(
            [
                "training_step",
                "group",
                "average_top20_cosine_similarity",
                "item_count",
            ]
        )


def append_trace(path, training_step, group_means, counts):
    with Path(path).open("a", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        for group in ("Head", "Tail"):
            writer.writerow(
                [
                    training_step,
                    group,
                    f"{group_means[group].mean().item():.8f}",
                    counts[group],
                ]
            )
