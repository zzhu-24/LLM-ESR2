#!/usr/bin/env python3
"""Compare learned ID and adapted-LLM item-neighbor similarities."""

import argparse
import csv
import html
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--id_checkpoint", required=True)
    parser.add_argument("--llm_checkpoint", required=True)
    parser.add_argument("--ts_item", required=True, type=int)
    parser.add_argument("--topk", default=20, type=int)
    parser.add_argument("--batch_size", default=512, type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--output_dir", default="outputs/item_neighbor_similarity"
    )
    parser.add_argument("--inter_file", default="inter")
    return parser.parse_args()


def load_state_dict(path):
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unsupported checkpoint format: {path}")
    return {
        key.removeprefix("module."): value.detach().cpu()
        for key, value in checkpoint.items()
    }


def load_item_popularity(dataset, inter_file):
    interaction_path = Path("data") / dataset / "handled" / f"{inter_file}.txt"
    pairs = []
    item_num = 0
    with interaction_path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            fields = line.rstrip().split()
            if len(fields) != 2:
                raise ValueError(
                    f"Expected '<user_id> <item_id>' at {interaction_path}:{line_number}."
                )
            item_id = int(fields[1])
            pairs.append(item_id)
            item_num = max(item_num, item_id)

    popularity = np.bincount(pairs, minlength=item_num + 1).astype(np.int64)
    return popularity, item_num


def get_id_embeddings(state_dict, item_num):
    if "item_emb.weight" not in state_dict:
        raise KeyError("ID checkpoint is missing item_emb.weight.")
    embeddings = state_dict["item_emb.weight"].float()
    if embeddings.shape[0] < item_num + 1:
        raise ValueError("ID checkpoint has fewer item rows than the dataset.")
    return embeddings[1:item_num + 1]


def get_adapted_llm_embeddings(state_dict, item_num):
    required_keys = (
        "llm_item_emb.weight",
        "adapter.0.weight",
        "adapter.0.bias",
        "adapter.1.weight",
        "adapter.1.bias",
    )
    missing_keys = [key for key in required_keys if key not in state_dict]
    if missing_keys:
        raise KeyError(f"LLM checkpoint is missing: {', '.join(missing_keys)}")

    llm_embeddings = state_dict["llm_item_emb.weight"].float()
    if llm_embeddings.shape[0] < item_num + 1:
        raise ValueError("LLM checkpoint has fewer item rows than the dataset.")
    llm_embeddings = llm_embeddings[1:item_num + 1]
    hidden = F.linear(
        llm_embeddings,
        state_dict["adapter.0.weight"].float(),
        state_dict["adapter.0.bias"].float(),
    )
    return F.linear(
        hidden,
        state_dict["adapter.1.weight"].float(),
        state_dict["adapter.1.bias"].float(),
    )


def mean_topk_by_group(embeddings, popularity, threshold, topk, batch_size, device):
    item_count = embeddings.shape[0]
    if item_count <= topk:
        raise ValueError(f"topk={topk} requires more than {topk} real items.")

    normalized = F.normalize(embeddings.to(device), p=2, dim=1)
    popularity = torch.as_tensor(popularity[1:item_count + 1], device=device)
    group_masks = {
        "Head": popularity >= threshold,
        "Tail": (popularity > 0) & (popularity < threshold),
    }
    sums = {
        group: torch.zeros(topk, dtype=torch.float64, device=device)
        for group in group_masks
    }
    counts = {group: 0 for group in group_masks}

    with torch.no_grad():
        for start in range(0, item_count, batch_size):
            end = min(start + batch_size, item_count)
            similarities = normalized[start:end] @ normalized.T
            rows = torch.arange(end - start, device=device)
            similarities[rows, rows + start] = -torch.inf
            top_values = torch.topk(
                similarities, k=topk, dim=1, largest=True, sorted=True
            ).values

            for group, mask in group_masks.items():
                batch_mask = mask[start:end]
                if torch.any(batch_mask):
                    sums[group] += top_values[batch_mask].double().sum(dim=0)
                    counts[group] += int(batch_mask.sum().item())

    for group, count in counts.items():
        if count == 0:
            raise ValueError(
                f"No {group.lower()} items for ts_item={threshold}; choose another threshold."
            )
    means = {group: (sums[group] / counts[group]).cpu().numpy() for group in sums}
    return means, counts


def write_csv(results, counts, output_path, topk):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=[
                "model",
                "group",
                "neighbor_rank",
                "average_cosine_similarity",
                "item_count",
            ],
        )
        writer.writeheader()
        for model, group_results in results.items():
            for group, values in group_results.items():
                for rank in range(1, topk + 1):
                    writer.writerow(
                        {
                            "model": model,
                            "group": group,
                            "neighbor_rank": rank,
                            "average_cosine_similarity": f"{values[rank - 1]:.8f}",
                            "item_count": counts[group],
                        }
                    )


def _polyline_points(values, x0, y0, width, height, y_min, y_max):
    x_step = width / max(1, len(values) - 1)
    y_span = max(y_max - y_min, 1e-8)
    return " ".join(
        f"{x0 + index * x_step:.2f},{y0 + height * (y_max - value) / y_span:.2f}"
        for index, value in enumerate(values)
    )


def write_svg(results, counts, dataset, threshold, output_path, topk):
    width, height = 1100, 460
    panel_width, panel_height = 430, 300
    panel_y = 85
    panel_xs = (80, 620)
    all_values = np.concatenate(
        [values for group_results in results.values() for values in group_results.values()]
    )
    y_min = float(all_values.min())
    y_max = float(all_values.max())
    margin = max((y_max - y_min) * 0.08, 0.01)
    y_min -= margin
    y_max += margin
    colors = {"Head": "#2563eb", "Tail": "#dc2626"}

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="30" text-anchor="middle" font-family="sans-serif" font-size="20">'
        f'{html.escape(dataset)} item Top-{topk} neighbor similarity (ts_item={threshold})</text>',
    ]

    for panel_x, (model, group_results) in zip(panel_xs, results.items()):
        svg.extend(
            [
                f'<line x1="{panel_x}" y1="{panel_y + panel_height}" x2="{panel_x + panel_width}" y2="{panel_y + panel_height}" stroke="#111827"/>',
                f'<line x1="{panel_x}" y1="{panel_y}" x2="{panel_x}" y2="{panel_y + panel_height}" stroke="#111827"/>',
                f'<text x="{panel_x + panel_width / 2}" y="65" text-anchor="middle" font-family="sans-serif" font-size="17">{html.escape(model)}</text>',
                f'<text x="{panel_x + panel_width / 2}" y="440" text-anchor="middle" font-family="sans-serif" font-size="13">Neighbor rank</text>',
            ]
        )
        for tick in (1, 5, 10, 15, topk):
            tick_x = panel_x + (tick - 1) * panel_width / max(1, topk - 1)
            svg.append(
                f'<text x="{tick_x:.2f}" y="410" text-anchor="middle" font-family="sans-serif" font-size="11">{tick}</text>'
            )
        for tick_index in range(5):
            tick_value = y_min + (y_max - y_min) * tick_index / 4
            tick_y = panel_y + panel_height * (y_max - tick_value) / (y_max - y_min)
            svg.extend(
                [
                    f'<line x1="{panel_x}" y1="{tick_y:.2f}" x2="{panel_x + panel_width}" y2="{tick_y:.2f}" stroke="#e5e7eb"/>',
                    f'<text x="{panel_x - 8}" y="{tick_y + 4:.2f}" text-anchor="end" font-family="sans-serif" font-size="11">{tick_value:.3f}</text>',
                ]
            )
        for group, values in group_results.items():
            points = _polyline_points(
                values, panel_x, panel_y, panel_width, panel_height, y_min, y_max
            )
            svg.append(
                f'<polyline points="{points}" fill="none" stroke="{colors[group]}" stroke-width="2.5"/>'
            )
        legend_x = panel_x + 245
        for offset, group in enumerate(("Head", "Tail")):
            legend_y = panel_y + 20 + offset * 22
            svg.extend(
                [
                    f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 26}" y2="{legend_y}" stroke="{colors[group]}" stroke-width="2.5"/>',
                    f'<text x="{legend_x + 34}" y="{legend_y + 4}" font-family="sans-serif" font-size="12">{group} (n={counts[group]})</text>',
                ]
            )

    svg.append(
        '<text x="18" y="235" text-anchor="middle" font-family="sans-serif" font-size="13" transform="rotate(-90 18 235)">Average cosine similarity</text>'
    )
    svg.append("</svg>")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(svg), encoding="utf-8")


def main():
    args = parse_args()
    if args.topk <= 0 or args.batch_size <= 0 or args.ts_item <= 0:
        raise ValueError("topk, batch_size, and ts_item must all be positive.")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    popularity, item_num = load_item_popularity(args.dataset, args.inter_file)
    id_embeddings = get_id_embeddings(load_state_dict(args.id_checkpoint), item_num)
    llm_embeddings = get_adapted_llm_embeddings(
        load_state_dict(args.llm_checkpoint), item_num
    )
    if id_embeddings.shape[1] != llm_embeddings.shape[1]:
        raise ValueError(
            "The ID and adapted LLM embeddings must have the same dimension: "
            f"{id_embeddings.shape[1]} != {llm_embeddings.shape[1]}."
        )

    results = {}
    counts = None
    for model_name, embeddings in (
        ("ID SASRec", id_embeddings),
        ("LLM embedding + adapter SASRec", llm_embeddings),
    ):
        group_means, group_counts = mean_topk_by_group(
            embeddings,
            popularity,
            args.ts_item,
            args.topk,
            args.batch_size,
            device,
        )
        results[model_name] = group_means
        counts = group_counts

    output_dir = Path(args.output_dir)
    output_stem = f"{args.dataset}_sasrec_top{args.topk}_neighbor_similarity"
    csv_path = output_dir / f"{output_stem}.csv"
    svg_path = output_dir / f"{output_stem}.svg"
    write_csv(results, counts, csv_path, args.topk)
    write_svg(results, counts, args.dataset, args.ts_item, svg_path, args.topk)
    print(f"CSV: {csv_path}")
    print(f"SVG: {svg_path}")


if __name__ == "__main__":
    main()
