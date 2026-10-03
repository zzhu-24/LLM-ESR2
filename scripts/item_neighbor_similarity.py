#!/usr/bin/env python3
"""Plot Head/Tail Top-20 neighbor similarity over optimizer steps."""

import argparse
import csv
import html
from pathlib import Path


TOPK = 20


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--id_trace", required=True)
    parser.add_argument("--llm_trace", required=True)
    parser.add_argument(
        "--output_dir", default="outputs/item_neighbor_similarity"
    )
    return parser.parse_args()


def load_trace(path):
    series = {"Head": [], "Tail": []}
    counts = {}
    with Path(path).open("r", encoding="utf-8") as input_file:
        for row in csv.DictReader(input_file):
            group = row["group"]
            series[group].append(
                (
                    int(row["training_step"]),
                    float(row["average_top20_cosine_similarity"]),
                )
            )
            counts[group] = int(row["item_count"])
    for group, values in series.items():
        if not values:
            raise ValueError(f"Trace {path} has no {group} rows.")
        values.sort()
    return series, counts


def _polyline_points(values, x0, y0, width, height, x_max, y_min, y_max):
    x_span = max(x_max, 1)
    y_span = max(y_max - y_min, 1e-8)
    return " ".join(
        f"{x0 + width * step / x_span:.2f},"
        f"{y0 + height * (y_max - value) / y_span:.2f}"
        for step, value in values
    )


def write_svg(results, counts, dataset, output_path):
    width, height = 1100, 460
    panel_width, panel_height = 430, 300
    panel_y = 85
    panel_xs = (80, 620)
    all_values = [
        value
        for group_results in results.values()
        for values in group_results.values()
        for _, value in values
    ]
    y_min = min(all_values)
    y_max = max(all_values)
    margin = max((y_max - y_min) * 0.08, 0.01)
    y_min -= margin
    y_max += margin
    colors = {"Head": "#2563eb", "Tail": "#dc2626"}

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="30" text-anchor="middle" font-family="sans-serif" font-size="20">'
        f'{html.escape(dataset)} item Top-{TOPK} neighbor similarity during training</text>',
    ]

    for panel_x, (model, group_results) in zip(panel_xs, results.items()):
        x_max = max(step for values in group_results.values() for step, _ in values)
        svg.extend(
            [
                f'<line x1="{panel_x}" y1="{panel_y + panel_height}" x2="{panel_x + panel_width}" y2="{panel_y + panel_height}" stroke="#111827"/>',
                f'<line x1="{panel_x}" y1="{panel_y}" x2="{panel_x}" y2="{panel_y + panel_height}" stroke="#111827"/>',
                f'<text x="{panel_x + panel_width / 2}" y="65" text-anchor="middle" font-family="sans-serif" font-size="17">{html.escape(model)}</text>',
                f'<text x="{panel_x + panel_width / 2}" y="440" text-anchor="middle" font-family="sans-serif" font-size="13">Training step</text>',
            ]
        )
        for tick_index in range(5):
            tick_step = round(x_max * tick_index / 4)
            tick_x = panel_x + panel_width * tick_index / 4
            svg.append(
                f'<text x="{tick_x:.2f}" y="410" text-anchor="middle" font-family="sans-serif" font-size="11">{tick_step}</text>'
            )
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
                values,
                panel_x,
                panel_y,
                panel_width,
                panel_height,
                x_max,
                y_min,
                y_max,
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
                    f'<text x="{legend_x + 34}" y="{legend_y + 4}" font-family="sans-serif" font-size="12">{group} (n={counts[model][group]})</text>',
                ]
            )

    svg.extend(
        [
            '<text x="18" y="235" text-anchor="middle" font-family="sans-serif" font-size="13" transform="rotate(-90 18 235)">Mean Top-20 cosine similarity</text>',
            "</svg>",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(svg), encoding="utf-8")


def main():
    args = parse_args()
    id_series, id_counts = load_trace(args.id_trace)
    llm_series, llm_counts = load_trace(args.llm_trace)
    results = {
        "ID SASRec": id_series,
        "LLM embedding + adapter SASRec": llm_series,
    }
    counts = {
        "ID SASRec": id_counts,
        "LLM embedding + adapter SASRec": llm_counts,
    }
    output_path = Path(args.output_dir) / (
        f"{args.dataset}_sasrec_top{TOPK}_training_steps.svg"
    )
    write_svg(results, counts, args.dataset, output_path)
    print(f"SVG: {output_path}")


if __name__ == "__main__":
    main()
