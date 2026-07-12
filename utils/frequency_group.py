import csv
import math
from html import escape
from pathlib import Path

import numpy as np


def parse_frequency_thresholds(thresholds):
    if thresholds is None or thresholds == "":
        return []
    if isinstance(thresholds, str):
        values = [int(part.strip()) for part in thresholds.split(",") if part.strip()]
    else:
        values = [int(value) for value in thresholds]
    values = sorted(set(values))
    if any(value <= 0 for value in values):
        raise ValueError("Frequency thresholds must be positive integers.")
    return values


def build_frequency_group_report(pred_rank, frequencies, topk=10, bin_size=1, thresholds=None):
    pred_rank = np.asarray(pred_rank)
    frequencies = np.asarray(frequencies).astype(np.int64)
    if pred_rank.shape[0] != frequencies.shape[0]:
        raise ValueError("pred_rank and frequencies must have the same length.")
    if pred_rank.shape[0] == 0:
        raise ValueError("Cannot build a frequency report from empty predictions.")
    if bin_size <= 0:
        raise ValueError("bin_size must be a positive integer.")

    thresholds = parse_frequency_thresholds(thresholds)
    groups = _threshold_groups(frequencies, thresholds) if thresholds else _fixed_width_groups(frequencies, bin_size)
    hr_key = f"HR@{topk}"
    ndcg_key = f"NDCG@{topk}"
    rows = []

    for label, lower, upper, mask in groups:
        count = int(np.sum(mask))
        if count == 0:
            continue

        ranks = pred_rank[mask]
        hit_mask = ranks < topk
        hr = float(np.sum(hit_mask) / count)
        ndcg = float(np.sum(np.where(hit_mask, 1 / np.log2(ranks + 2), 0.0)) / count)
        rows.append({
            "group": label,
            "freq_min": int(lower),
            "freq_max": "" if upper is None else int(upper),
            "count": count,
            hr_key: hr,
            ndcg_key: ndcg,
        })

    if not rows:
        raise ValueError("All frequency groups are empty.")
    return rows


def write_frequency_group_csv(rows, output_path, topk=10):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["group", "freq_min", "freq_max", "count", f"HR@{topk}", f"NDCG@{topk}"]
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_frequency_group_svg(rows, output_path, title, x_label, count_label, topk=10):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    width, height = 1120, 640
    margin_left, margin_right, margin_top, margin_bottom = 86, 92, 78, 108
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    chart_bottom = height - margin_bottom
    chart_right = width - margin_right
    hr_key = f"HR@{topk}"
    ndcg_key = f"NDCG@{topk}"
    max_count = max(row["count"] for row in rows) or 1
    max_metric = max(max(row[hr_key], row[ndcg_key]) for row in rows)
    metric_max = max(0.05, min(1.0, math.ceil(max_metric * 20) / 20))
    metric_max = 1.0 if metric_max > 0.95 else metric_max

    slot_width = plot_width / len(rows)
    bar_width = max(1, slot_width * 0.66)
    label_every = max(1, math.ceil(len(rows) / 18))

    elems = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{margin_left}" y="36" font-family="Arial, sans-serif" font-size="22" font-weight="700" fill="#111827">{escape(title)}</text>',
        f'<text x="{margin_left}" y="60" font-family="Arial, sans-serif" font-size="13" fill="#4b5563">Bars: {escape(count_label)}. Lines: SASRec {escape(hr_key)} and {escape(ndcg_key)}.</text>',
        f'<line x1="{margin_left}" y1="{chart_bottom}" x2="{chart_right}" y2="{chart_bottom}" stroke="#374151" stroke-width="1"/>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{chart_bottom}" stroke="#374151" stroke-width="1"/>',
        f'<line x1="{chart_right}" y1="{margin_top}" x2="{chart_right}" y2="{chart_bottom}" stroke="#374151" stroke-width="1"/>',
    ]

    for tick in range(6):
        count_value = max_count * tick / 5
        y = chart_bottom - (count_value / max_count) * plot_height
        elems.append(f'<line x1="{margin_left}" y1="{y:.1f}" x2="{chart_right}" y2="{y:.1f}" stroke="#e5e7eb" stroke-width="1"/>')
        elems.append(f'<text x="{margin_left - 10}" y="{y + 4:.1f}" text-anchor="end" font-family="Arial, sans-serif" font-size="12" fill="#4b5563">{_format_count_tick(count_value)}</text>')

        metric_value = metric_max * tick / 5
        metric_y = chart_bottom - (metric_value / metric_max) * plot_height
        elems.append(f'<text x="{chart_right + 10}" y="{metric_y + 4:.1f}" font-family="Arial, sans-serif" font-size="12" fill="#4b5563">{metric_value:.2f}</text>')

    centers = []
    for idx, row in enumerate(rows):
        x_center = margin_left + slot_width * idx + slot_width / 2
        centers.append(x_center)
        bar_height = (row["count"] / max_count) * plot_height
        x = x_center - bar_width / 2
        y = chart_bottom - bar_height
        elems.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" fill="#4f83cc" opacity="0.82"/>')
        if idx % label_every == 0 or idx == len(rows) - 1:
            elems.append(
                f'<text x="{x_center:.1f}" y="{chart_bottom + 20}" transform="rotate(-38 {x_center:.1f} {chart_bottom + 20})" '
                f'text-anchor="end" font-family="Arial, sans-serif" font-size="11" fill="#4b5563">{escape(row["group"])}</text>'
            )

    elems.extend(_line_elements(rows, centers, hr_key, metric_max, chart_bottom, plot_height, "#15803d", "HR"))
    elems.extend(_line_elements(rows, centers, ndcg_key, metric_max, chart_bottom, plot_height, "#dc2626", "NDCG"))

    legend_x = chart_right - 240
    legend_y = margin_top - 18
    elems.extend([
        f'<rect x="{legend_x}" y="{legend_y - 13}" width="12" height="12" fill="#4f83cc" opacity="0.82"/>',
        f'<text x="{legend_x + 18}" y="{legend_y - 3}" font-family="Arial, sans-serif" font-size="12" fill="#111827">{escape(count_label)}</text>',
        f'<line x1="{legend_x + 114}" y1="{legend_y - 7}" x2="{legend_x + 142}" y2="{legend_y - 7}" stroke="#15803d" stroke-width="2.5"/>',
        f'<text x="{legend_x + 148}" y="{legend_y - 3}" font-family="Arial, sans-serif" font-size="12" fill="#111827">{escape(hr_key)}</text>',
        f'<line x1="{legend_x + 190}" y1="{legend_y - 7}" x2="{legend_x + 218}" y2="{legend_y - 7}" stroke="#dc2626" stroke-width="2.5"/>',
        f'<text x="{legend_x + 224}" y="{legend_y - 3}" font-family="Arial, sans-serif" font-size="12" fill="#111827">{escape(ndcg_key)}</text>',
        f'<text x="{width / 2}" y="{height - 24}" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" fill="#111827">{escape(x_label)}</text>',
        f'<text x="24" y="{height / 2}" transform="rotate(-90 24 {height / 2})" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" fill="#111827">{escape(count_label)}</text>',
        f'<text x="{width - 24}" y="{height / 2}" transform="rotate(90 {width - 24} {height / 2})" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" fill="#111827">Performance</text>',
        "</svg>",
    ])
    output_path.write_text("\n".join(elems))


def _fixed_width_groups(frequencies, bin_size):
    groups = []
    if np.any(frequencies == 0):
        groups.append(("0", 0, 0, frequencies == 0))

    positive = frequencies[frequencies > 0]
    if positive.size == 0:
        return groups

    max_freq = int(np.max(positive))
    for idx in range(math.ceil(max_freq / bin_size)):
        lower = idx * bin_size + 1
        upper = min((idx + 1) * bin_size, max_freq)
        label = str(lower) if lower == upper else f"{lower}-{upper}"
        groups.append((label, lower, upper, (frequencies >= lower) & (frequencies <= upper)))
    return groups


def _threshold_groups(frequencies, thresholds):
    groups = []
    previous = None
    for threshold in thresholds:
        if previous is None:
            label = f"<{threshold}"
            mask = frequencies < threshold
            groups.append((label, 0, threshold - 1, mask))
        else:
            label = f"{previous}-{threshold - 1}"
            mask = (frequencies >= previous) & (frequencies < threshold)
            groups.append((label, previous, threshold - 1, mask))
        previous = threshold

    label = f">={thresholds[-1]}"
    groups.append((label, thresholds[-1], None, frequencies >= thresholds[-1]))
    return groups


def _line_elements(rows, centers, metric_key, metric_max, chart_bottom, plot_height, color, label):
    points = []
    for row, x in zip(rows, centers):
        y = chart_bottom - (row[metric_key] / metric_max) * plot_height
        points.append((x, y))

    elems = [
        '<polyline fill="none" stroke="{color}" stroke-width="2.6" stroke-linejoin="round" stroke-linecap="round" points="{points}"/>'.format(
            color=color,
            points=" ".join(f"{x:.1f},{y:.1f}" for x, y in points),
        )
    ]
    if len(points) <= 80:
        for x, y in points:
            elems.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="#ffffff" stroke="{color}" stroke-width="1.8"><title>{escape(label)} {metric_key}</title></circle>')
    return elems


def _format_count_tick(value):
    if value >= 1000000:
        return f"{value / 1000000:.1f}M"
    if value >= 1000:
        return f"{value / 1000:.1f}K"
    return str(int(round(value)))
