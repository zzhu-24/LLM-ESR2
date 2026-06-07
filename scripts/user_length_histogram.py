import argparse
import math
from collections import Counter, defaultdict
from pathlib import Path


def read_user_lengths(inter_file):
    user_items = defaultdict(int)
    with inter_file.open("r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            user_items[parts[0]] += 1
    return list(user_items.values())


def make_bins(lengths, bin_size):
    max_len = max(lengths)
    bin_count = math.ceil(max_len / bin_size)
    bins = Counter(((length - 1) // bin_size) for length in lengths)
    return [(idx * bin_size + 1, min((idx + 1) * bin_size, max_len), bins[idx]) for idx in range(bin_count)]


def write_svg_histogram(bins, output, title):
    width, height = 980, 560
    margin_left, margin_right, margin_top, margin_bottom = 70, 28, 58, 78
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    max_count = max(count for _, _, count in bins) or 1
    bar_gap = 2
    bar_width = max(1, (plot_width - bar_gap * (len(bins) - 1)) / len(bins))

    elems = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{margin_left}" y="34" font-family="Arial, sans-serif" font-size="22" font-weight="700" fill="#111827">{title}</text>',
        f'<line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - margin_right}" y2="{height - margin_bottom}" stroke="#374151" stroke-width="1"/>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height - margin_bottom}" stroke="#374151" stroke-width="1"/>',
    ]

    for tick in range(6):
        value = max_count * tick / 5
        y = height - margin_bottom - (value / max_count) * plot_height
        elems.append(f'<line x1="{margin_left - 5}" y1="{y:.1f}" x2="{width - margin_right}" y2="{y:.1f}" stroke="#e5e7eb" stroke-width="1"/>')
        elems.append(f'<text x="{margin_left - 10}" y="{y + 4:.1f}" text-anchor="end" font-family="Arial, sans-serif" font-size="12" fill="#4b5563">{int(value)}</text>')

    label_every = max(1, math.ceil(len(bins) / 18))
    for idx, (start, end, count) in enumerate(bins):
        x = margin_left + idx * (bar_width + bar_gap)
        bar_height = (count / max_count) * plot_height
        y = height - margin_bottom - bar_height
        elems.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" fill="#2563eb"/>')
        if idx % label_every == 0 or idx == len(bins) - 1:
            label = str(start) if start == end else f"{start}-{end}"
            elems.append(f'<text x="{x + bar_width / 2:.1f}" y="{height - margin_bottom + 20}" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#4b5563">{label}</text>')

    elems.extend([
        f'<text x="{width / 2}" y="{height - 22}" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" fill="#111827">User history length</text>',
        f'<text x="18" y="{height / 2}" transform="rotate(-90 18 {height / 2})" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" fill="#111827">User count</text>',
        "</svg>",
    ])
    output.write_text("\n".join(elems))


def main():
    parser = argparse.ArgumentParser(description="Report user sequence length stats and draw a histogram.")
    parser.add_argument("--dataset", default="beauty2014", help="dataset folder under data/")
    parser.add_argument("--inter_file", default=None, help="path to inter.txt; overrides --dataset")
    parser.add_argument("--bin_size", default=1, type=int, help="histogram bin size")
    parser.add_argument("--output", default=None, help="output SVG path")
    args = parser.parse_args()

    inter_file = Path(args.inter_file) if args.inter_file else Path("data") / args.dataset / "handled" / "inter.txt"
    if not inter_file.exists():
        raise FileNotFoundError(f"Missing interaction file: {inter_file}")

    lengths = read_user_lengths(inter_file)
    if not lengths:
        raise ValueError(f"No user interactions found in {inter_file}")

    output = Path(args.output) if args.output else Path("outputs") / f"{args.dataset}_user_length_histogram.svg"
    output.parent.mkdir(parents=True, exist_ok=True)

    bins = make_bins(lengths, args.bin_size)
    write_svg_histogram(bins, output, f"{args.dataset} user history length distribution")

    avg_len = sum(lengths) / len(lengths)
    sorted_lengths = sorted(lengths)
    n = len(sorted_lengths)
    p90_idx = int(n * 0.9)
    p95_idx = int(n * 0.95)
    p99_idx = int(n * 0.99)
    p90 = sorted_lengths[p90_idx]
    p95 = sorted_lengths[p95_idx]
    p99 = sorted_lengths[p99_idx]
    print(f"dataset: {args.dataset}")
    print(f"users: {len(lengths)}")
    print(f"min_len: {min(lengths)}")
    print(f"avg_len: {avg_len:.4f}")
    print(f"max_len: {max(lengths)}")
    print(f"p90_len: {p90}")
    print(f"p95_len: {p95}")
    print(f"p99_len: {p99}")
    print(f"histogram: {output}")


if __name__ == "__main__":
    main()