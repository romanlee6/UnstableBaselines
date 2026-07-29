#!/usr/bin/env bash
set -euo pipefail

python -u tests/test_multirole_transactional_train.py
python -u third_party/TextArena/tests/test_ub_collector_flow.py
python -u third_party/TextArena/tests/test_ipd_predict.py
python -u tests/test_ipd_no_predict_matched.py
python -u tests/test_phase_balanced_reinforce.py
