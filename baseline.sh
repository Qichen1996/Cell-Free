#!/usr/bin/env bash
# run_baseline.sh
ACC=3000                # 加速倍率
SCN=B                   # traffic 场景
AGENT=fixed             # baseline：Always-On
W_QOS=40                # 这里只是留着占位，fixed 实际不会用到

for SEED in 1 2 3; do
  LOG="logs/sim_${AGENT}_s${SEED}.log"
  echo ">> running ${AGENT}  seed=${SEED}  log=${LOG}"
  nohup ./simulate.py \
        -A  ${AGENT} \
        -S  ${SCN} \
        --w_qos  ${W_QOS} \
        --seed   ${SEED} \
        -a       ${ACC} \
        --experiment_name "${AGENT}_seed${SEED}" \
        --group_name      "${AGENT}" \
        > "${LOG}" 2>&1 &
done
