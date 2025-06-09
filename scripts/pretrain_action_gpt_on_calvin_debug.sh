export CUDA_VISIBLE_DEVICES=0
export PROJECT_ROOT='/workspace/APT'

cd ${PROJECT_ROOT}
if [ ! -f .project-root ]; then
    touch .project-root
fi

cd ${PROJECT_ROOT}/action_gpt/train
accelerate launch --main_process_port 29501 train_action_gpt.py --config_path "${PROJECT_ROOT}/action_gpt/configs/train/debug.yaml"

<<COMMENT
nohup bash pretrain_action_gpt_on_calvin_debug.sh > pretrain_action_gpt_on_calvin_debug.log 2>&1 &
tail -f pretrain_action_gpt_on_calvin_debug.log
COMMENT