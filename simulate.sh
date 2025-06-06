#!/bin/sh
acc=3000
MODEL_BASE="/home/jovyan/LKY-TEST/Cell-Free/results/MultiCellNetwork"
RUN_NAME="90%gain_step10"  # 你的 wandb 目录名

for seed in 1 2 3; do
for S in B; do
    for w in 40; do
        MODEL_DIR="${MODEL_BASE}/${S}/mappo/check/wandb/${RUN_NAME}/files"
        ./simulate.py -S $S --w_qos $w --model_dir "$MODEL_DIR" --seed $seed -a $acc $@
        # ./simulate.py -S $S --w_qos $w --seed $seed -a $acc $@
        # ./simulate.py -S $S -A fixed --seed $seed  --model_dir "$MODEL_DIR" -a $acc $@
    done
    # --model_dir "$MODEL_DIR"
    # w=40
    # MODEL_DIR="${MODEL_BASE}/${S}/mappo/check/wandb/${RUN_NAME}/files"
    #./simulate.py -S $S --w_qos $w --seed $seed -a $acc $@
    # ./simulate.py -S $S -A fixed --w_qos $w --seed $seed  --model_dir "$MODEL_DIR" -a $acc $@
    # ./simulate.py -S $S -A simple1 --seed $seed -a $acc $@
    # ./simulate.py -S $S -A simple1 --no_offload --seed $seed -a $acc $@
    # for max_s in 1; do  # max depth of sleep
    #     # MODEL_DIR="${MODEL_BASE}/${S}/mappo/check/wandb/${RUN_NAME}/files"
    #     ./simulate.py -S $S --w_qos $w --seed $seed -a $acc --max_sleep $max_s $@
    # done
    # ./simulate.py -S $S --w_qos $w --seed $seed -a $acc --no_interf $@
    # ./simulate.py -S $S --w_qos $w --seed $seed -a $acc --no_offload $@
    # ./simulate.py -S $S -A dqn --seed $seed -a $acc --use_wandb --run_version run28 $@
    # ./simulate.py -S $S -A dqn --w_qos 4 --seed $seed -a $acc --use_wandb --run_version run23 --no_offload $@
done
done