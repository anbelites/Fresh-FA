"""Корневые пути проекта."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VIDEO_DIR = PROJECT_ROOT / "01.Video"
TRANSCRIPT_DIR = PROJECT_ROOT / "02.Transcript"
EVALUATION_DIR = PROJECT_ROOT / "03.Evaluation"
META_DIR = PROJECT_ROOT / "04.Meta"
CONFIG_DIR = PROJECT_ROOT / "config"
# Legacy: при импорте из config/ читался этот файл; slug в БД — без расширения.
CRITERIA_FILE = CONFIG_DIR / "criteria.yaml"
DEFAULT_CHECKLIST_SLUG = "criteria"
MANAGERS_FILE = CONFIG_DIR / "managers.yaml"
LOCATIONS_FILE = CONFIG_DIR / "locations.yaml"


def evaluation_json_path(stem: str, criteria_path: Path) -> Path:
    """Файл LLM-оценки: stem + slug чеклиста (criteria_path.name — только имя)."""
    return EVALUATION_DIR / f"{stem}__{criteria_path.name}.eval.json"


def human_evaluation_json_path(stem: str, criteria_name: str) -> Path:
    """Файл ручной (человеческой) оценки для пары (stem, имя чеклиста)."""
    return EVALUATION_DIR / f"{stem}__{criteria_name}.human.json"


def meta_json_path(stem: str) -> Path:
    """Per-video metadata sidecar."""
    return META_DIR / f"{stem}.meta.json"
