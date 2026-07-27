#!/usr/bin/env bash
set -euo pipefail

python -u third_party/TextArena/tests/test_ipd_predict.py
python -u third_party/TextArena/tests/test_ub_collector_flow.py
python -u tests/test_multirole_transactional_train.py
python -u tests/inspect_ipd_context.py
python -u tests/analyze_ipd_training_lengths.py
