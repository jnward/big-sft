#!/usr/bin/env bash
# Set up a fresh box to run the adapter eval pipeline.
#
# Idempotent: safe to re-run; skips work that's already done.
#
# Steps:
#   1. apt: docker.io, docker-compose-v2, ninja-build, python3.12-dev
#   2. /etc/docker/daemon.json with expanded default-address-pools
#   3. submodules: third_party/{harbor, terminal-wrench}
#   4. .venvs/vllm  (vllm 0.9.2 + transformers 4.51.3 + flash-attn)
#   5. .venvs/eval  (training pins from EVAL_HANDOFF + judge deps)
#   6. third_party/harbor uv sync
#   7. smoke check: harbor oracle agent on one task
#
# Pinned versions are intentional; see scripts/eval/CLAUDE.md "Pitfalls".

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO_ROOT"

log() { echo -e "\n=== $* ==="; }

# ----- 1. apt deps ---------------------------------------------------------
log "1/7  apt deps"
need_apt=()
for pkg in docker.io docker-compose-v2 ninja-build; do
  if ! dpkg -l "$pkg" >/dev/null 2>&1; then
    need_apt+=("$pkg")
  fi
done
if (( ${#need_apt[@]} )); then
  sudo apt-get update -y
  sudo apt-get install -y "${need_apt[@]}"
else
  echo "all apt deps already installed"
fi

# ----- 2. docker daemon config ---------------------------------------------
log "2/7  docker network pool config"
DAEMON_JSON=/etc/docker/daemon.json
need_restart=0
desired='{"default-address-pools":[{"base":"172.30.0.0/15","size":24},{"base":"172.32.0.0/15","size":24}]}'
if [[ ! -f "$DAEMON_JSON" ]] || ! grep -q '172.30.0.0/15' "$DAEMON_JSON" 2>/dev/null; then
  echo "$desired" | sudo tee "$DAEMON_JSON" >/dev/null
  need_restart=1
fi
if (( need_restart )); then
  sudo systemctl restart docker
fi
# socket access without docker group membership
if [[ -e /var/run/docker.sock ]] && ! [[ -w /var/run/docker.sock ]]; then
  sudo chmod 666 /var/run/docker.sock
fi
docker version --format '{{.Server.Version}}' >/dev/null

# ----- 3. submodules -------------------------------------------------------
log "3/7  submodules"
git submodule update --init --recursive
# wrench .git is 1.3GB of trajectory history we don't need; reclaim with a
# tree-filtered re-clone (working tree unchanged).
WRENCH_GIT=third_party/terminal-wrench/.git
if [[ -d "$WRENCH_GIT" ]] && [[ "$(du -sm "$WRENCH_GIT" | cut -f1)" -gt 200 ]]; then
  echo "shrinking wrench .git via partial clone..."
  url=$(git -C third_party/terminal-wrench remote get-url origin)
  rev=$(git -C third_party/terminal-wrench rev-parse HEAD)
  rm -rf third_party/terminal-wrench
  git clone --filter=tree:0 "$url" third_party/terminal-wrench
  git -C third_party/terminal-wrench checkout "$rev"
fi

# ----- 4. vllm venv --------------------------------------------------------
log "4/7  .venvs/vllm  (vllm + flash-attn + transformers 4.51.3)"
VLLM_VENV=.venvs/vllm
if [[ ! -x "$VLLM_VENV/bin/vllm" ]]; then
  uv venv "$VLLM_VENV" --python 3.12
  VIRTUAL_ENV="$VLLM_VENV" uv pip install "vllm==0.9.2"
  # vllm 0.9.2 conflicts with transformers >= 4.54 (aimv2 registration)
  VIRTUAL_ENV="$VLLM_VENV" uv pip install "transformers==4.51.3"
  VIRTUAL_ENV="$VLLM_VENV" uv pip install wheel packaging
  VIRTUAL_ENV="$VLLM_VENV" uv pip install "flash-attn==2.8.3" --no-build-isolation
else
  echo "$VLLM_VENV/bin/vllm exists; skipping"
fi

# ----- 5. eval/training venv -----------------------------------------------
log "5/7  .venvs/eval  (training pins + merge tooling + judge deps)"
EVAL_VENV=.venvs/eval
if [[ ! -d "$EVAL_VENV" ]]; then
  uv venv "$EVAL_VENV" --python 3.12
  VIRTUAL_ENV="$EVAL_VENV" uv pip install \
    "torch==2.5.1" --extra-index-url https://download.pytorch.org/whl/cu124
  VIRTUAL_ENV="$EVAL_VENV" uv pip install \
    "transformers==5.5.4" "trl==1.2.0" "accelerate==1.13.0" \
    "datasets==4.8.4" "peft==0.19.1" \
    wandb hf_transfer \
    "openai>=1.99" "python-dotenv" "tiktoken"
  VIRTUAL_ENV="$EVAL_VENV" uv pip install wheel packaging
  VIRTUAL_ENV="$EVAL_VENV" uv pip install "flash-attn==2.8.3" --no-build-isolation
  VIRTUAL_ENV="$EVAL_VENV" uv pip install safetensors
else
  echo "$EVAL_VENV exists; skipping"
fi

# ----- 6. harbor venv ------------------------------------------------------
log "6/7  harbor venv (uv sync inside third_party/harbor)"
( cd third_party/harbor && uv sync )

# ----- 7. smoke ------------------------------------------------------------
log "7/7  smoke: oracle agent on one task"
mkdir -p build/eval-dataset build/jobs build/logs build/merged
"$EVAL_VENV/bin/python" -m scripts.eval.prep_eval_dataset >/dev/null
JOB=$(date +smoke-%Y%m%d-%H%M%S)
( cd "$REPO_ROOT" && uv run --project third_party/harbor harbor run \
    -p build/eval-dataset -i 1018 -a oracle \
    -n 1 -k 1 --job-name "$JOB" --jobs-dir build/jobs )

log "setup complete"
echo "Next: drop adapter checkpoints under tb-eval/checkpoints/ and run:"
echo "  source .venvs/eval/bin/activate"
echo "  python -m scripts.eval.run --checkpoint <path> --mode both --job-name <name>"
