#!/bin/bash
# CausalFlow Pre-commit Test Script

set -e

echo "--- [CausalFlow CI] Starting Pre-commit Checks ---"

# 1. Build Rust core for Python
echo "[1/2] Rebuilding Rust extension..."
source .venv/bin/activate
PATH="/Users/ystk/.cargo/bin:$PATH" maturin develop

# 2. Run Python tests
echo "[2/2] Running Python tests with pytest..."
pytest py-causalflow/tests/test_flow.py

echo "--- [CausalFlow CI] All checks passed! Safe to commit. ---"
