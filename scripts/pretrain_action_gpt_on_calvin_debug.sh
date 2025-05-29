export TOKENIZERS_PARALLELISM=false
export TF_CPP_MIN_LOG_LEVEL=2

export CUDA_VISIBLE_DEVICES=0
export PROJECT_ROOT='/workspace/ActionGPT'

cd ${PROJECT_ROOT}
if [ ! -f .project-root ]; then
    touch .project-root
fi

cd ${PROJECT_ROOT}/action_gpt/train
accelerate launch --main_process_port 29501 train_action_gpt.py --config_path "${PROJECT_ROOT}/action_gpt/configs/train/data_calvin_debug-model_action_continuous.yaml"

<<COMMENT
nohup bash pretrain_action_gpt_on_calvin_debug.sh > pretrain_action_gpt_on_calvin_debug.log 2>&1 &
tail -f pretrain_action_gpt_on_calvin_debug.log
COMMENT