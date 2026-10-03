import csv
from pathlib import Path

import torch
import torch.nn.functional as F


def get_model_item_embeddings(model, model_name, item_num):
    """Return the current item representation used immediately before SASRec."""
    model = model.module if hasattr(model, "module") else model
    if model_name == "sasrec":
        return model.item_emb.weight[1:item_num + 1].detach()
    if model_name == "llm_adapter_sasrec":
        llm_embeddings = model.llm_item_emb.weight[1:item_num + 1]
        return model.adapter(llm_embeddings).detach()
    raise ValueError(f"Unsupported model for sequence tracking: {model_name}")


def mean_sequence_similarity_by_group(
    embeddings,
    popularity,
    threshold,
    training_sequences,
    max_len,
    batch_size,
):
    """Average current-item-to-history cosine similarity by popularity group."""
    device = embeddings.device
    normalized = F.normalize(embeddings, p=2, dim=1)
    padding = torch.zeros(
        (1, normalized.shape[1]), dtype=normalized.dtype, device=device
    )
    embedding_lookup = torch.cat([padding, normalized], dim=0)
    popularity = torch.as_tensor(popularity, device=device)
    sums = {"Head": 0.0, "Tail": 0.0}
    counts = {"Head": 0, "Tail": 0}

    with torch.no_grad():
        for start in range(0, len(training_sequences), batch_size):
            batch_sequences = training_sequences[start:start + batch_size]
            history_ids = torch.zeros(
                (len(batch_sequences), max_len), dtype=torch.long
            )
            current_ids = torch.empty(len(batch_sequences), dtype=torch.long)

            for row, sequence in enumerate(batch_sequences):
                if not sequence:
                    raise ValueError("Encountered an empty training sequence.")
                current_ids[row] = sequence[-1]
                history = sequence[:-1][-max_len:]
                if history:
                    history_ids[row, -len(history):] = torch.as_tensor(history)

            history_ids = history_ids.to(device)
            current_ids = current_ids.to(device)
            history_mask = history_ids > 0
            history_lengths = history_mask.sum(dim=1)
            valid_samples = history_lengths > 0
            if not torch.any(valid_samples):
                continue

            history_embeddings = embedding_lookup[history_ids]
            current_embeddings = embedding_lookup[current_ids]
            similarities = (
                history_embeddings * current_embeddings.unsqueeze(1)
            ).sum(dim=-1)
            sample_means = (
                (similarities * history_mask).sum(dim=1)
                / history_lengths.clamp_min(1)
            )

            group_masks = {
                "Head": (popularity[current_ids] >= threshold) & valid_samples,
                "Tail": (popularity[current_ids] < threshold) & valid_samples,
            }
            for group, mask in group_masks.items():
                if torch.any(mask):
                    sums[group] += sample_means[mask].double().sum().item()
                    counts[group] += int(mask.sum().item())

    for group, count in counts.items():
        if count == 0:
            raise ValueError(
                f"No valid {group.lower()} training samples for ts_item={threshold}."
            )
    return {group: sums[group] / counts[group] for group in sums}, counts


def initialize_trace(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(
            [
                "training_step",
                "group",
                "average_sequence_cosine_similarity",
                "sample_count",
            ]
        )


def append_trace(path, training_step, group_means, counts):
    with Path(path).open("a", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        for group in ("Head", "Tail"):
            writer.writerow(
                [training_step, group, f"{group_means[group]:.8f}", counts[group]]
            )
