"""Code to evaluate Calvin with Moto-GPT + ActionGPT hybrid approach."""
import pyrootutils
import os
import sys
pyrootutils.setup_root(__file__, indicator='.project-root', pythonpath=True, dotenv=True)

from common.models.model_utils import load_action_gpt_policy, load_moto_gpt_policy

from omegaconf import OmegaConf
import hydra
import argparse
import json
import numpy as np
import logging
from pathlib import Path
import time
import copy
from moviepy.editor import ImageSequenceClip
from accelerate import Accelerator
from datetime import timedelta
from accelerate.utils import InitProcessGroupKwargs

from calvin_agent.evaluation.multistep_sequences import get_sequences
from calvin_agent.evaluation.utils import (
    count_success,
    get_env_state_for_initial_condition,
    get_log_dir,
)
from calvin_utils import print_and_save

from termcolor import colored
import torch
from tqdm.auto import tqdm
from transformers import set_seed

logger = logging.getLogger(__name__)

os.environ["FFMPEG_BINARY"] = "auto-detect"
CALVIN_ROOT = os.environ['CALVIN_ROOT']

def make_env(dataset_path, observation_space, device, show_gui=False):
    val_folder = Path(dataset_path) / "training"
    from calvin_env_wrapper_raw import CalvinEnvWrapperRaw
    env = CalvinEnvWrapperRaw(val_folder, observation_space, device, show_gui)
    return env

class HybridPolicyWrapper:
    """Wrapper that combines Moto-GPT and ActionGPT policies."""
    
    def __init__(self, moto_gpt_policy, action_gpt_policy, switch_step=20):
        self.moto_gpt_policy = moto_gpt_policy
        self.action_gpt_policy = action_gpt_policy
        self.switch_step = switch_step
        self.current_step = 0
        
        # Store original test_chunk_sizes from config
        self.moto_chunk_size = 5  # Moto-GPT actual chunk_size from config
        self.action_chunk_size = 5  # ActionGPT default
        
    def reset(self):
        """Reset both policies and step counter."""
        self.current_step = 0
        self.moto_gpt_policy.reset()
        self.action_gpt_policy.reset()
        
    def reset_buffer(self):
        """Reset buffers for both policies."""
        if hasattr(self.moto_gpt_policy, 'reset_buffer'):
            self.moto_gpt_policy.reset_buffer()
        self.action_gpt_policy.reset_buffer()
        
    def step(self, obs, goal):
        """Step function that switches between policies based on current step."""
        if self.current_step < self.switch_step:
            # Use Moto-GPT for first 20 steps
            
            action = self.moto_gpt_policy.step(obs, goal)
            
            # Update ActionGPT's prev_action_buffer with Moto-GPT's action
            if hasattr(self.action_gpt_policy, 'prev_action_buffer'):
                # Convert action to appropriate format and update buffer
                if isinstance(action, np.ndarray):
                    action_tensor = torch.from_numpy(action).float()
                else:
                    action_tensor = action.float()
                    
                if action_tensor.dim() == 1:
                    action_tensor = action_tensor.unsqueeze(0)  # Add batch dim if needed
                
                # Get the number of actions to add (both have chunk_size=5, so should be compatible)
                num_actions = min(action_tensor.shape[0], self.action_gpt_policy.prev_action_buffer.shape[1])
                
                # Update ActionGPT's buffer by shifting and adding new actions
                self.action_gpt_policy.prev_action_buffer = torch.cat([
                    self.action_gpt_policy.prev_action_buffer[:, num_actions:],
                    action_tensor[:num_actions].unsqueeze(0)
                ], dim=1)
        else:
            # Use ActionGPT after switch_step
            start_time = time.time()

            action = self.action_gpt_policy.step(obs, goal)
            inference_time = time.time() - start_time
            print(f"inference time: {inference_time:.4f} seconds")
            
        self.current_step += 1
        return action

def evaluate_policy(moto_gpt_model, action_gpt_model, env, eval_sr_path, eval_result_path, ep_len, num_sequences, num_procs, procs_id, eval_dir=None, debug=False, switch_step=20):
    conf_dir = Path(f"{CALVIN_ROOT}/calvin_models") / "conf"
    task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)
    val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")
    eval_dir = get_log_dir(eval_dir)
    eval_sequences = get_sequences(num_sequences)
    num_seq_per_procs = num_sequences // num_procs
    eval_sequences = eval_sequences[num_seq_per_procs*procs_id:num_seq_per_procs*(procs_id+1)]

    # Create hybrid policy wrapper
    hybrid_model = HybridPolicyWrapper(moto_gpt_model, action_gpt_model, switch_step)

    results = []
    if not debug:
        eval_sequences = tqdm(eval_sequences, position=0, leave=True)

    sequence_i = 0
    for initial_state, eval_sequence in eval_sequences:
        result = evaluate_sequence(env, hybrid_model, task_oracle, initial_state, eval_sequence, val_annotations, debug, eval_dir, sequence_i, ep_len)
        results.append(result)
        if not debug:
            success_list = count_success(results)
            with open(eval_sr_path, 'a') as f:
                line =f"{sequence_i}/{num_sequences}: "
                for sr in success_list:
                    line += f"{sr:.3f} | "
                sequence_i += 1
                line += "\n"
                f.write(line)
            eval_sequences.set_description(
                " ".join([f"{i + 1}/5 : {v * 100:.1f}% |" for i, v in enumerate(success_list)]) + "|"
            )
        else:
            sequence_i += 1
    print_and_save(results, eval_sequences, eval_result_path, None)
    return results

def evaluate_sequence(env, model, task_checker, initial_state, eval_sequence, val_annotations, debug, eval_dir, sequence_i, ep_len):
    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
    model.reset_buffer()  # buffer reset
    success_counter = 0
    if debug:
        time.sleep(1)
        print()
        print()
        print(f"Evaluating sequence: {' -> '.join(eval_sequence)}")
        print("Subtask: ", end="")
    for subtask_i, subtask in enumerate(eval_sequence):
        success = rollout(env, model, task_checker, subtask, val_annotations, debug, eval_dir, subtask_i, sequence_i, ep_len)
        if success:
            success_counter += 1
        else:
            return success_counter
    return success_counter

def rollout(env, model, task_oracle, subtask, val_annotations, debug, eval_dir, subtask_i, sequence_i, ep_len):
    if debug:
        print(f"{subtask} ", end="")
        time.sleep(0.5)
    obs = env.get_obs()
    lang_annotation = val_annotations[subtask][0]
    model.reset()  # Reset the hybrid model for each subtask
    start_info = env.get_info()
    if debug:
        img_list = []
    unfinished = 0
    
    for step in range(ep_len):
        if unfinished == 0:
            action = model.step(obs, lang_annotation)
            unfinished = action.shape[0]
        obs, _, _, current_info = env.step(action[-unfinished])
        unfinished -= 1
        if debug:
            img_copy = copy.deepcopy(obs['rgb_obs']['rgb_static'])
            img_list.append(img_copy)
        # check if current step solves a task
        current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
        if len(current_task_info) > 0:
            if debug:
                print(colored("success", "green"), end=" ")
                clip = ImageSequenceClip(img_list, fps=30)
                clip.write_gif(os.path.join(eval_dir, f'{sequence_i}-{subtask_i}-{subtask}-succ.gif'), fps=30)
            return True
    if debug:
        print(colored("fail", "red"), end=" ")
        clip = ImageSequenceClip(img_list, fps=30)
        clip.write_gif(os.path.join(eval_dir, f'{sequence_i}-{subtask_i}-{subtask}-fail.gif'), fps=30)
    return False

def main(args):
    print(args)
    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=3600))
    acc = Accelerator(kwargs_handlers=[kwargs])
    device = acc.device
    
    # Load ActionGPT
    args.is_gripper_binary = True
    action_gpt_eva = load_action_gpt_policy(args)
    action_gpt_eva.policy = acc.prepare(action_gpt_eva.policy, device_placement=[True])
    action_gpt_eva.policy.eval()
    
    # Load Moto-GPT with its own test_chunk_size
    moto_args = argparse.Namespace(**vars(args))
    moto_args.test_chunk_size = 5  # Moto-GPT original default test_chunk_size
    moto_gpt_eva = load_moto_gpt_policy(moto_args)
    moto_gpt_eva.policy = acc.prepare(moto_gpt_eva.policy, device_placement=[True])
    moto_gpt_eva.policy.eval()

    # Prepare CALVIN Environment
    observation_space = {
        'rgb_obs': ['rgb_static'], 
        'depth_obs': [], 
        'state_obs': ['robot_obs'], 
        'actions': ['rel_actions'], 
        'language': ['language']}

    try:
        eval_dir = os.path.join(args.eval_dir, f'eval{torch.cuda.current_device()}/')
    except:
        eval_dir = os.path.join(args.eval_dir, 'eval')
    os.makedirs(eval_dir, exist_ok=True)
    env = make_env('fake_dataset', observation_space, device, show_gui=args.show_gui)
    acc.print(f"initialize CALVIN environment")

    # Evaluation with hybrid approach
    avg_reward = torch.tensor(evaluate_policy(
        moto_gpt_eva, 
        action_gpt_eva,
        env,
        os.path.join(args.eval_dir,'success_rate.txt'),
        os.path.join(args.eval_dir,'result.txt'),
        args.ep_len,
        args.num_sequences,
        acc.num_processes,
        acc.process_index,
        eval_dir,
        debug=args.record_evaluation_video,
        switch_step=args.switch_step
    )).float().mean().to(device)
    acc.wait_for_everyone()
    avg_reward = acc.gather_for_metrics(avg_reward).mean()
    if acc.is_main_process:
        print('average success rate ', avg_reward)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--action_gpt_path', type=str, required=True)
    parser.add_argument('--moto_gpt_path', type=str, required=True)
    parser.add_argument('--test_chunk_size', type=int, default=5)
    parser.add_argument('--switch_step', type=int, default=20)
    
    # Moto-GPT specific arguments with their original default values
    parser.add_argument('--mask_latent_motion_probability', type=float, default=1.0)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--sample', type=str, default='true')  # Will be converted to bool
    parser.add_argument('--top_k', type=int, default=0)
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--beam_size', type=int, default=5)
    parser.add_argument('--parallel', type=str, default='false')  # Will be converted to bool
    parser.add_argument('--use_temporal_ensemble', type=str, default='false')  # Will be converted to bool

    parser.add_argument('--num_sequences', type=int, default=1000)
    parser.add_argument('--ep_len', type=int, default=360)
    parser.add_argument('--eval_dir', type=str, required=True)

    parser.add_argument('--show_gui', action='store_true')
    parser.add_argument('--record_evaluation_video', action='store_true')

    args = parser.parse_args()
    
    # Convert string boolean arguments to actual booleans
    args.sample = args.sample.lower() == 'true'
    args.parallel = args.parallel.lower() == 'true'
    args.use_temporal_ensemble = args.use_temporal_ensemble.lower() == 'true'
    
    set_seed(12345)  # Use Moto-GPT's original seed
    main(args)