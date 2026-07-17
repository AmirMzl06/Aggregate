from pathlib import Path

_current_dir = Path(__file__).parent.resolve()

PROJECT_ROOT_DIR = _current_dir.parent.resolve()
CEBRA_DIR = (PROJECT_ROOT_DIR / "CEBRA-main").resolve()
CEBRA_Parallel_DIR = (PROJECT_ROOT_DIR / "CEBRA_Parallel").resolve()
DATA_DIR = (PROJECT_ROOT_DIR / "data").resolve()
