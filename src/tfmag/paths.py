from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RepoPaths:
    root: Path
    data_raw: Path
    data_processed: Path
    outputs: Path
    outputs_figures: Path
    outputs_tables: Path
    outputs_logs: Path
    configs: Path


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_paths() -> RepoPaths:
    root = get_repo_root()
    data_raw = root / "data" / "raw"
    data_processed = root / "data" / "processed"
    outputs = root / "outputs"
    outputs_figures = outputs / "figures"
    outputs_tables = outputs / "tables"
    outputs_logs = outputs / "logs"
    configs = root / "configs"
    return RepoPaths(
        root=root,
        data_raw=data_raw,
        data_processed=data_processed,
        outputs=outputs,
        outputs_figures=outputs_figures,
        outputs_tables=outputs_tables,
        outputs_logs=outputs_logs,
        configs=configs,
    )


def ensure_base_dirs() -> RepoPaths:
    paths = get_paths()
    for path in [
        paths.data_raw,
        paths.data_processed,
        paths.outputs_figures,
        paths.outputs_tables,
        paths.outputs_logs,
        paths.configs,
    ]:
        path.mkdir(parents=True, exist_ok=True)
    return paths
