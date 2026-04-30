from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetPaths:
    root: Path
    handled: Path
    inter_file: Path
    llm_item_emb: Path
    id_item_emb: Path
    sim_users: Path


def get_dataset_paths(dataset: str, inter_file: str = "inter") -> DatasetPaths:
    root = Path("data") / dataset
    handled = root / "handled"
    return DatasetPaths(
        root=root,
        handled=handled,
        inter_file=handled / f"{inter_file}.txt",
        llm_item_emb=handled / "itm_emb_np.pkl",
        id_item_emb=handled / "pca64_itm_emb_np.pkl",
        sim_users=handled / "sim_user_100.pkl",
    )
