

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
        self.prev_action_buffer = torch.zeros(1, self.prev_action_buffer_size, self.act_dim)  # (1, buffer_size, act_dim)

    def step(self, obs, goal):
        """Step function."""
        # Language
        lang_inputs = self.lang_tokenizer(goal, return_tensors='pt', padding=True)
        tokenized_text = lang_inputs.input_ids
        lang_attention_mask = lang_inputs.attention_mask

        # RGB
        rgb = self.rgb_process(obs['rgb_obs']['rgb_static'])
        # print(f"RGB after processing: {rgb.shape}")
        rgb_data = rgb.unsqueeze(0)

        # # Attention mask
        attention_mask = torch.ones(1, self.seq_len).long()
        
        # Forward pass
        tokenized_text = tokenized_text.to(self.device)
        lang_attention_mask = lang_attention_mask.to(self.device) if lang_attention_mask is not None else None
        rgb_data = rgb_data.to(self.device)
        attention_mask = attention_mask.to(self.device)
        rgb = rgb.to(self.device)
        prev_actions = self.prev_action_buffer.to(self.device)

        with torch.no_grad():
            prediction = self.policy(
                rgb=rgb_data, 
                language=tokenized_text,
                attention_mask=attention_mask,
                prev_actions=prev_actions,
                lang_attention_mask=lang_attention_mask,
        )

        # Arm action
        arm_action_preds = prediction['arm_action_preds']  # (1, t, chunk_size, act_dim - 1)
        if self.pred_discrete_arm_action:
            arm_action_preds = arm_action_preds.view(-1, self.act_dim - 1, 3)
        else:
            arm_action_preds = arm_action_preds.view(-1, self.act_dim - 1)  # (t*chunk_size, act_dim - 1)
        
        # Gripper action
        gripper_action_preds = prediction['gripper_action_preds']
        gripper_action_preds = gripper_action_preds.view(-1, 1)  # (t*chunk_size, 1)

        # Use the first test_chunk_size action
        arm_action_pred = arm_action_preds[:self.test_chunk_size]  # (test_chunk_size, act_dim - 1)
        gripper_action_pred = gripper_action_preds[:self.test_chunk_size]  # (test_chunk_size, 1)
        
        if self.is_gripper_binary:
            gripper_action_pred = ((gripper_action_pred > 0).float()) * 2.0 - 1.0
        
        if self.pred_discrete_arm_action:
            arm_action_pred = arm_action_pred.softmax(dim=-1).argmax(dim=-1)
            
        action_pred = torch.cat((arm_action_pred, gripper_action_pred), dim=-1)  # (test_chunk_size, act_dim)
        action_pred = action_pred.detach().cpu()

        excuted_action = action_pred
        # Update prev action buffer
        self.prev_action_buffer = torch.cat([
            self.prev_action_buffer[:, 1:],  # (1, 9, action_dim)
            excuted_action.unsqueeze(0)  # (1, 1, actiom_dim)
        ], dim=1)  # (1, 10, ction_dim)
        
        self.rollout_step_counter += 1
        return action_pred  # (1, action_dim)