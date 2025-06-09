#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export PROJECT_ROOT='/home/kaurml/APT'
export CALVIN_ROOT=${PROJECT_ROOT}/../calvin/
export ACTION_GPT_PATH="${PROJECT_ROOT}/action_gpt/checkpoints/action_gpt_pretrained_on_calvin"
export MOTO_GPT_PATH="/home/kaurml/Moto/moto_gpt/checkpoints/moto_gpt_finetuned_on_calvin"
export EVAL_DIR="${PROJECT_ROOT}/action_gpt/evaluation/calvin/eval_results_hybrid"

# Add Moto-GPT to Python path
export PYTHONPATH="/home/kaurml/Moto:$PYTHONPATH"

export TEST_CHUNK_SIZE=5
export SWITCH_STEP=10

EvalCALVIN_Hybrid() {
cd ${PROJECT_ROOT}/action_gpt/evaluation/calvin
accelerate launch evaluate_calvin.py \
    --action_gpt_path ${ACTION_GPT_PATH} \
    --moto_gpt_path ${MOTO_GPT_PATH} \
    --test_chunk_size ${TEST_CHUNK_SIZE} \
    --switch_step ${SWITCH_STEP} \
    --eval_dir ${EVAL_DIR}
    # --show_gui
    # --record_evaluation_video \

echo "Done! EvalCALVIN_Hybrid ${EVAL_DIR}"
}
EvalCALVIN_Hybrid

<<COMMENT
ps aux | grep 'evaluate_calvin' | awk '{print $2}' | xargs kill -9
cd ${PROJECT_ROOT}/scripts
nohup bash evaluate_calvin_hybrid.sh > evaluate_calvin_hybrid.log 2>&1 &
tail -f evaluate_calvin_hybrid.log
COMMENT