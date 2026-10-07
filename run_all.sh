#!/bin/bash
# Runs every step in order. About 12 minutes on a laptop; the base model is trained once and kept in base.pt.
set -e
cd "$(dirname "$0")"
for f in step1_base step2_reward step3_reinforce step4_group step5_clip step6_train step7_what_changed step8_knobs exercise_cost; do
  echo "=== $f.py"
  python3 $f.py
done
