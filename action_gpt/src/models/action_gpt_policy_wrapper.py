# action_gpt_policy_wrapper.py
import torch
import torchvision.transforms as T
import numpy as np
from PIL import Image
from common.processors.preprocessor_utils import get_rgb_preprocessor
import torch.nn.functional as F

class ActionGPT_PolicyWrapper:
    def __init__(
            self,
            policy,
            variant,
            lang_tokenizer
    ):
        """Constructor."""
        self.test_chunk_size = variant['test_chunk_size']
        self.is_gripper_binary = variant['is_gripper_binary']
        self.pred_discrete_arm_action = variant['pred_discrete_arm_action']
        self.lang_tokenizer = lang_tokenizer

        # RGB Preprocess
        input_size = variant['rgb_shape']
        rgb_mean = variant['rgb_mean']
        rgb_std = variant['rgb_std']
        self.transform = T.Compose([
            T.Resize(input_size, interpolation=Image.BICUBIC),
            T.Normalize(rgb_mean, rgb_std)
        ])

        self.policy = policy
        self.act_dim = variant['act_dim']
        self.seq_len = variant['seq_len']
        self.chunk_size = variant['chunk_size']
        self.prev_action_buffer_size = variant['prev_action_buffer_size']
        
        self.is_prev_action_buffer = variant.get('is_prev_action_buffer', True)
        self.use_robot_obs = variant.get('use_robot_obs', False)
        self.use_gripper_rgb = variant.get('use_gripper_rgb', False)
        
        self.valid_prev_actions_count = 0
        self.rollout_step_counter = 0
        
    @property
    def device(self):
        return self.policy.device

    def rgb_process(self, rgb):
        rgb = Image.fromarray(rgb)
        rgb = T.ToTensor()(rgb.convert('RGB'))
        rgb = self.transform(rgb)
        return rgb
        
    def reset(self):
        """Reset function."""
        self.rollout_step_counter = 0

    def reset_buffer(self):
        """Reset function."""
        if self.is_prev_action_buffer:
            self.prev_action_buffer = torch.zeros(1, self.prev_action_buffer_size, self.act_dim)

    def add_action_to_buffer(self, action):
        """외부에서 action을 buffer에 추가"""
        if not self.is_prev_action_buffer:
            return
            
        if not hasattr(self, 'prev_action_buffer'):
            self.reset_buffer()
        
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action)
        
        if action.dim() == 1:
            action = action.unsqueeze(0)
        if action.dim() == 2 and action.shape[0] > 1:
            action = action[:1]
            
        self.prev_action_buffer = torch.cat([
            self.prev_action_buffer[:, 1:],
            action.unsqueeze(0)
        ], dim=1)

    def step(self, obs, goal):
        """Step function."""
        if self.is_prev_action_buffer and not hasattr(self, 'prev_action_buffer'):
            self.reset_buffer()
            
        # Language processing
        lang_inputs = self.lang_tokenizer(goal, return_tensors='pt', padding=True)
        tokenized_text = lang_inputs.input_ids
        lang_attention_mask = lang_inputs.attention_mask

        # Static RGB processing
        rgb_static = self.rgb_process(obs['rgb_obs']['rgb_static'])
        rgb_static = rgb_static.unsqueeze(0).unsqueeze(0)  # (1, 1, c, h, w)
        
        # Gripper RGB processing
        if self.use_gripper_rgb and 'rgb_gripper' in obs['rgb_obs']:
            rgb_gripper = self.rgb_process(obs['rgb_obs']['rgb_gripper'])
            rgb_gripper = rgb_gripper.unsqueeze(0).unsqueeze(0)  # (1, 1, c, h, w)
        else:
            rgb_gripper = None
        
        # Forward pass
        tokenized_text = tokenized_text.to(self.device)
        lang_attention_mask = lang_attention_mask.to(self.device) if lang_attention_mask is not None else None
        rgb_static = rgb_static.to(self.device)
        if rgb_gripper is not None:
            rgb_gripper = rgb_gripper.to(self.device)
        
        # prev_actions 전달
        if self.is_prev_action_buffer:
            prev_actions = self.prev_action_buffer.to(self.device)
        else:
            prev_actions = None
        
        # robot_obs 전달
        if self.use_robot_obs and 'robot_obs' in obs:
            robot_obs = torch.tensor(obs['robot_obs']).float().unsqueeze(0).to(self.device)  # (1, robot_obs_dim)
        else:
            robot_obs = None
        
        with torch.no_grad():
            prediction = self.policy(
                rgb_static=rgb_static,        # static RGB
                language=tokenized_text,
                rgb_gripper=rgb_gripper,      # gripper RGB
                robot_obs=robot_obs,          # robot observation
                prev_actions=prev_actions,    # prev actions
                lang_attention_mask=lang_attention_mask,
        )

        # Arm action
        arm_action_preds = prediction['arm_action_preds']  # (1, t, chunk_size, act_dim - 1)
        arm_action_preds = arm_action_preds.view(-1, self.act_dim - 1)  # (t*chunk_size, act_dim - 1)
        
        # Gripper action
        gripper_action_preds = prediction['gripper_action_preds']
        gripper_action_preds = gripper_action_preds.view(-1, 1)  # (t*chunk_size, 1)

        # Use the first test_chunk_size action
        arm_action_pred = arm_action_preds[:self.test_chunk_size]  # (test_chunk_size, act_dim - 1)
        gripper_action_pred = gripper_action_preds[:self.test_chunk_size]  # (test_chunk_size, 1)
        
        if self.is_gripper_binary:
            gripper_action_pred = ((gripper_action_pred > 0).float()) * 2.0 - 1.0
            
        action_pred = torch.cat((arm_action_pred, gripper_action_pred), dim=-1)  # (test_chunk_size, act_dim)
        action_pred = action_pred.detach().cpu()

        if self.is_prev_action_buffer:
            self.prev_action_buffer = torch.cat([
                self.prev_action_buffer[:, self.test_chunk_size:],
                action_pred.unsqueeze(0)
            ], dim=1)
          
        self.rollout_step_counter += 1
        return action_pred