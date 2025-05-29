export CUDA_VISIBLE_DEVICES=0,1,2,3
export PROJECT_ROOT='/workspace/ActionGPT'

cd ${PROJECT_ROOT}/action_gpt/train
accelerate launch --main_process_port 29501 train_action_gpt.py --config_path "${PROJECT_ROOT}/action_gpt/configs/train/data_calvin_task_abc_d-model_action_continuous.yaml"

<<COMMENT
nohup bash pretrain_action_gpt_on_calvin_task_abc_d.sh > pretrain_action_gpt_on_calvin_task_abc_d.log 2>&1 &
tail -f pretrain_action_gpt_on_calvin_task_abc_d.log
COMMENT