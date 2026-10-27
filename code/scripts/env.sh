# Shared setup sourced by the launchers: repo root, Python environment and settings.
#   REPO    repo root, derived from this file's location (the launcher cds there)
#   VENV    virtualenv to activate (default: $REPO/.venv; skipped if it does not exist)
#   CONFIG  settings file whose `export` lines are loaded (default: $REPO/config.env, untracked;
#           copy config.env.example). Only `export` lines are read, nothing else in the file runs.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO}"
VENV="${VENV:-${REPO}/.venv}"
if [[ -f "${VENV}/bin/activate" ]]; then source "${VENV}/bin/activate"; fi
CONFIG="${CONFIG:-${REPO}/config.env}"
if [[ ! -f "${CONFIG}" ]]; then
  echo "ERROR: settings file not found: ${CONFIG} (copy config.env.example to config.env)" >&2
  exit 1
fi
eval "$(grep '^export ' "${CONFIG}")"
