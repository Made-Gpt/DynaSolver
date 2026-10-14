# Source from the repo root:  source scripts/env.sh
# Makes flat imports (models / layers / data_provider / ...) resolvable.
_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${_REPO_ROOT}:${_REPO_ROOT}/dynasolver${PYTHONPATH:+:$PYTHONPATH}"
echo "PYTHONPATH updated for DynaSolver (${_REPO_ROOT})"
unset _REPO_ROOT
