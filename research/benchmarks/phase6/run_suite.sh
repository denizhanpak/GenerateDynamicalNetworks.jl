#!/bin/bash
# Run the Phase 6 suite in batches under 10 minutes until every case is done (or 40 batches).
cd "$(dirname "$0")/../../notebooks"
log=${GDN_LOG:-../benchmarks/phase6/results/suite.log}
for b in $(seq 1 40); do
  timeout -s KILL 590 julia --project=. -t 7 ../benchmarks/phase6/suite.jl 240 480 >> "$log" 2>&1
  code=$?
  echo "batch $b exit $code: $(tail -1 "$log")"
  [ $code -eq 0 ] && grep -q "finished: 13 of 13" <(tail -1 "$log") && break
  [ $code -ne 0 ] && [ $code -ne 3 ] && [ $code -ne 137 ] && break
done
