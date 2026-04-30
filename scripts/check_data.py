import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.paths import get_dataset_paths


def read_inter_stats(path):
    user_num = 0
    item_num = 0
    interaction_count = 0
    with path.open("r") as f:
        for line in f:
            user, item = line.rstrip().split(" ")
            user_num = max(user_num, int(user))
            item_num = max(item_num, int(item))
            interaction_count += 1
    return user_num, item_num, interaction_count


def read_pickle_shape(path):
    with path.open("rb") as f:
        value = pickle.load(f)
    return np.asarray(value).shape


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="yelp")
    parser.add_argument("--inter_file", default="inter")
    args = parser.parse_args()

    paths = get_dataset_paths(args.dataset, args.inter_file)
    required_files = [
        paths.inter_file,
        paths.llm_item_emb,
        paths.id_item_emb,
        paths.sim_users,
    ]

    missing = [path for path in required_files if not path.exists()]
    if missing:
        print("Missing files:")
        for path in missing:
            print(f"  {path}")
        raise SystemExit(1)

    user_num, item_num, interaction_count = read_inter_stats(paths.inter_file)
    llm_shape = read_pickle_shape(paths.llm_item_emb)
    id_shape = read_pickle_shape(paths.id_item_emb)
    sim_shape = read_pickle_shape(paths.sim_users)

    print(f"dataset: {args.dataset}")
    print(f"users: {user_num}")
    print(f"items: {item_num}")
    print(f"interactions: {interaction_count}")
    print(f"llm item embedding shape: {llm_shape}")
    print(f"id item embedding shape: {id_shape}")
    print(f"similar users shape: {sim_shape}")

    if llm_shape[0] not in {item_num, item_num + 1, item_num + 2}:
        print("Warning: llm item embedding row count does not match item count.")
    if id_shape[0] not in {item_num, item_num + 1, item_num + 2}:
        print("Warning: id item embedding row count does not match item count.")
    if sim_shape[0] != user_num:
        print("Warning: similar-user row count does not match user count.")


if __name__ == "__main__":
    main()
