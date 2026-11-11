# Source this inside WSL before running anything:  source scripts/wsl_env.sh
# Code lives on the Windows checkout; the venv, HF cache and datasets live on WSL's
# ext4 disk (whose vhdx is itself on F:). See DECISIONS.md D-001.
export RETRACE_ROOT="/mnt/f/Github/retrace"
export RETRACE_DATA="${RETRACE_DATA:-$HOME/retrace-data}"
export UV_PROJECT_ENVIRONMENT="$HOME/.venvs/retrace"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export PATH="$HOME/.local/bin:$PATH"
mkdir -p "$RETRACE_DATA"
cd "$RETRACE_ROOT"
