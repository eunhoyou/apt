export CUDA_VISIBLE_DEVICES=0
export PROJECT_ROOT='/home/kaurml/APT'
export CALVIN_ROOT=${PROJECT_ROOT}/../calvin/
export ACTION_GPT_PATH="${PROJECT_ROOT}/action_gpt/checkpoints/action_gpt_pretrained_on_calvin"
export EVAL_DIR="${PROJECT_ROOT}/action_gpt/evaluation/calvin/eval_results"

export TEST_CHUNK_SIZE=1

EvalCALVIN() {
cd ${PROJECT_ROOT}/action_gpt/evaluation/calvin
accelerate launch evaluate_calvin.py \
    --action_gpt_path ${ACTION_GPT_PATH} \
    --test_chunk_size ${TEST_CHUNK_SIZE} \
    --eval_dir ${EVAL_DIR} \
    # --show_gui
    # --record_evaluation_video \

    
echo "Done! EvalCALVIN ${EVAL_DIR}"
}
EvalCALVIN

<<COMMENT
ps aux | grep 'evaluate_calvin' | awk '{print $2}' | xargs kill -9
cd ${PROJECT_ROOT}/scripts
nohup bash evaluate_calvin.sh > evaluate_calvin.log 2>&1 &
tail -f evaluate_calvin.log
COMMENT
