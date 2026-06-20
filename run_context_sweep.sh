#!/usr/bin/env bash
# Context x P2P sweep: runs context_p2p.scenarios.yaml once per input length,
# each into its own result subdir, by overriding random_input_len via --bench-set.
# Then plot_context_curve.py stitches them into the TTFT-vs-context curve.
set -euo pipefail
SPEC="${SPEC:-config/5060ti-16gb/4-cards/_pending/context_p2p.scenarios.yaml}"
RUN="${RUN:-results/$(date +%F)_5060ti-16gb_4cards_gen3-x4_context-p2p}"
LENGTHS="${LENGTHS:-1024 8192 32768 131072 245760}"   # all < 262144 incl. output
P2P_BIN="${P2P_BIN:-}"

for L in $LENGTHS; do
  echo "=== context sweep: random_input_len=$L ==="
  python -m p2pbench --scenarios "$SPEC" \
      --output-dir "$RUN/ctx_$L" \
      ${P2P_BIN:+--p2p-test-bin "$P2P_BIN"} \
      --bench-set "random_input_len=$L" \
      run
done
python tools/plot_context_curve.py "$RUN"
echo "Done. Curve + CSV under $RUN/"
