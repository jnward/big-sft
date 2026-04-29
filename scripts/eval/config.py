"""Single source of truth for paths and shared constants.

All eval scripts import from here; nothing should hardcode `/workspace/...`.
"""
from __future__ import annotations

import os
from pathlib import Path

# Repo layout
REPO_ROOT = Path(__file__).resolve().parents[2]
THIRD_PARTY = REPO_ROOT / "third_party"
HARBOR_DIR = THIRD_PARTY / "harbor"
WRENCH_DIR = THIRD_PARTY / "terminal-wrench"
WRENCH_TASKS_DIR = WRENCH_DIR / "tasks"
WRENCH_MONITOR_DIR = WRENCH_DIR / "monitoring"
WRENCH_INDEX_DIR = WRENCH_DIR / "index"

# Eval prompts (vendored copies)
PROMPTS_DIR = REPO_ROOT / "scripts" / "eval" / "prompts"

# Build artifacts (gitignored)
BUILD_DIR = REPO_ROOT / "build"
EVAL_DATASET_DIR = BUILD_DIR / "eval-dataset"
JOBS_DIR = BUILD_DIR / "jobs"
MERGED_DIR = BUILD_DIR / "merged"
LOGS_DIR = BUILD_DIR / "logs"
VLLM_PID_FILE = BUILD_DIR / "vllm.pid"

# Held-out eval task split
TASK_SPLIT_PATH = REPO_ROOT / "tb-eval" / "data" / "task_split.json"

# Adapter scale formula constants. Mirror scripts/gr/adapter.py:78-87.
# scale = lora_alpha * sqrt(1 / (f_nonlin * d))   (match_rslora=True branch)
ADAPTER_LORA_ALPHA = 32
ADAPTER_F_NONLIN = 0.5
ADAPTER_D_RETAIN = 64
ADAPTER_D_FORGET = 64

# Eval defaults
DEFAULT_AGENT_TIMEOUT_SEC = 360.0
DEFAULT_MAX_TURNS = 64
DEFAULT_N_CONCURRENT = 64
DEFAULT_VLLM_PORT = 8000
DEFAULT_MAX_MODEL_LEN = 16384
DEFAULT_GPU_MEMORY_UTIL = 0.92
DEFAULT_DP_SIZE = 8
DEFAULT_BASE_MODEL = "Qwen/Qwen3-32B"
DEFAULT_JUDGE_MODEL = "openai/gpt-5.4-nano"
DEFAULT_JUDGE_PROMPT = "judge_v3"

# OpenRouter is the canonical judge endpoint when OPENAI_BASE_URL isn't set explicitly.
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def env_path(name: str, default: Path) -> Path:
    """Allow operator to override paths via env var without editing code."""
    v = os.environ.get(name)
    return Path(v) if v else default
