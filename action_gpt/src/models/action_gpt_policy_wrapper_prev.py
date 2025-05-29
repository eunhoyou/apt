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

        # Preprocess
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
        self.prev_action_buffer_size = variant['prev_action_buffer_size']  # variant.get('prev_action_buffer_size', 10)
        self.use_temporal_ensemble = variant['use_temporal_ensemble']  # variant.get('use_temporal_ensemble', False)
        
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
        
        self.prev_action_buffer = torch.zeros(1, self.prev_action_buffer_size, self.act_dim)

        if self.use_temporal_ensemble:
            if self.pred_discrete_arm_action:
                self.action_buffer = np.zeros((self.test_chunk_size, self.test_chunk_size, (self.act_dim-1)*3+1))
            else:
                self.action_buffer = np.zeros((self.test_chunk_size, self.test_chunk_size, self.act_dim))
            self.action_buffer_mask = np.zeros((self.test_chunk_size, self.test_chunk_size), dtype=bool)

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
        # attention_mask = torch.ones(1, self.seq_len).long()
        
        # Forward pass
        tokenized_text = tokenized_text.to(self.device)
        lang_attention_mask = lang_attention_mask.to(self.device) if lang_attention_mask is not None else None
        rgb_data = rgb_data.to(self.device)
        # attention_mask = attention_mask.to(self.device)
        rgb = rgb.to(self.device)
        prev_actions = self.prev_action_buffer.to(self.device)

        with torch.no_grad():
            prediction = self.policy(
                rgb=rgb_data, 
                language=tokenized_text,
                # attention_mask=attention_mask,
                prev_actions=prev_actions,
                train=False,
                lang_attention_mask=lang_attention_mask,
                in_simulation=True,
        )

        # Arm action
        arm_action_preds = prediction['arm_action_preds']  # (1, t, chunk_size, act_dim - 1)
        if self.pred_discrete_arm_action:
            arm_action_preds = arm_action_preds.view(-1, self.act_dim - 1, 3)
        else:
            arm_action_preds = arm_action_preds.view(-1, self.act_dim - 1)  # (t*chunk_size, act_dim - 1)
        
        # Gripper action
        gripper_action_preds = prediction['gripper_action_preds']  # (1, t, chunk_size, 1)
        gripper_action_preds = gripper_action_preds.view(-1, 1)  # (t*chunk_size, 1)
        

        # Use the first test_chunk_size action
        arm_action_pred = arm_action_preds[:self.test_chunk_size]  # (test_chunk_size, act_dim - 1)
        gripper_action_pred = gripper_action_preds[:self.test_chunk_size]  # (test_chunk_size, 1)
        
        if not self.use_temporal_ensemble:
            if self.is_gripper_binary:
                gripper_action_pred = ((gripper_action_pred > 0).float()) * 2.0 - 1.0
            
            if self.pred_discrete_arm_action:
                arm_action_pred = arm_action_pred.softmax(dim=-1).argmax(dim=-1)
                
            action_pred = torch.cat((arm_action_pred, gripper_action_pred), dim=-1)  # (test_chunk_size, act_dim)
            action_pred = action_pred.detach().cpu()
        else:
            # Shift action buffer
            self.action_buffer[1:, :, :] = self.action_buffer[:-1, :, :]
            self.action_buffer_mask[1:, :] = self.action_buffer_mask[:-1, :]
            self.action_buffer[:, :-1, :] = self.action_buffer[:, 1:, :]
            self.action_buffer_mask[:, :-1] = self.action_buffer_mask[:, 1:]
            self.action_buffer_mask = self.action_buffer_mask * np.fliplr(np.triu(np.ones(self.test_chunk_size))).astype(bool)

            # Add to action buffer
            if self.pred_discrete_arm_action:
                action = torch.cat((arm_action_pred.reshape(arm_action_pred.shape[0], -1), gripper_action_pred), dim=-1) # (t*chunk_size, (act_dim-1)*3+1)
            else:
                action = torch.cat((arm_action_pred, gripper_action_pred), dim=-1) # (t*chunk_size, act_dim)
            action = action.detach().cpu().numpy()
            self.action_buffer[0] = action
            self.action_buffer_mask[0] = True
            
            # Ensemble temporally to predict action
            action_pred = np.sum(self.action_buffer[:, 0, :] * self.action_buffer_mask[:, 0:1], axis=0) / np.sum(self.action_buffer_mask[:, 0], axis=0)
            action_pred = torch.from_numpy(action_pred)

            # Make gripper action either -1 or 1
            if self.is_gripper_binary:
                action_pred[-1] = 1 if action_pred[-1] > 0 else -1
            
            if self.pred_discrete_arm_action:
                arm_action_pred = action_pred[:-1]
                arm_action_pred = arm_action_pred.reshape(-1, 3)
                arm_action_pred = arm_action_pred.softmax(dim=-1).argmax(dim=-1)
                action_pred = torch.cat([arm_action_pred, action_pred[-1:]], dim=-1)
            
            action_pred = action_pred.reshape(1, self.act_dim)

        # Update prev action buffer
        flattened_actions = action_pred[0].reshape(-1, self.act_dim).detach().cpu()
        total_new_actions = flattened_actions.shape[0]
        if total_new_actions >= self.prev_action_buffer_size:
            self.prev_action_buffer = flattened_actions[-self.prev_action_buffer_size:].unsqueeze(0)
        else:
            self.prev_action_buffer = torch.cat([
                self.prev_action_buffer[:, total_new_actions:], 
                flattened_actions.unsqueeze(0)
            ], dim=1)
        
        self.rollout_step_counter += 1
        return action_pred
    

# import torch
# import torchvision.transforms as T
# import numpy as np
# from PIL import Image
# from common.processors.preprocessor_utils import get_rgb_preprocessor
# import torch.nn.functional as F

# class ActionGPT_PolicyWrapper:
#     def __init__(
#             self,
#             policy,
#             variant,
#             lang_tokenizer
#     ):
#         """Constructor."""
#         self.test_chunk_size = variant['test_chunk_size']
#         self.is_gripper_binary = variant['is_gripper_binary']
#         self.pred_discrete_arm_action = variant['pred_discrete_arm_action']
#         self.lang_tokenizer = lang_tokenizer

#         # Preprocess
#         input_size = variant['rgb_shape']
#         rgb_mean = variant['rgb_mean']
#         rgb_std = variant['rgb_std']
#         self.transform = T.Compose([
#             T.Resize(input_size, interpolation=Image.BICUBIC),
#             T.Normalize(rgb_mean, rgb_std)
#         ])

#         self.policy = policy
        
#         self.act_dim = variant['act_dim']
#         self.seq_len = variant['seq_len']
#         self.chunk_size = variant['chunk_size']
#         self.prev_action_buffer_size = variant['prev_action_buffer_size']
#         self.use_temporal_ensemble = variant['use_temporal_ensemble']
        
#     @property
#     def device(self):
#         return self.policy.device

#     def rgb_process(self, rgb):
#         rgb = Image.fromarray(rgb)
#         rgb = T.ToTensor()(rgb.convert('RGB'))
#         rgb = self.transform(rgb)
#         return rgb
        
#     def reset(self):
#         """Reset function."""
#         self.rollout_step_counter = 0
        
#         self.prev_action_buffer = torch.zeros(1, self.prev_action_buffer_size, self.act_dim)

#         if self.use_temporal_ensemble:
#             if self.pred_discrete_arm_action:
#                 self.action_buffer = np.zeros((self.test_chunk_size, self.test_chunk_size, (self.act_dim-1)*3+1))
#             else:
#                 self.action_buffer = np.zeros((self.test_chunk_size, self.test_chunk_size, self.act_dim))
#             self.action_buffer_mask = np.zeros((self.test_chunk_size, self.test_chunk_size), dtype=bool)

#     def step(self, obs, goal):
#         """Step function."""
#         # Language
#         lang_inputs = self.lang_tokenizer(goal, return_tensors='pt', padding=True)
#         tokenized_text = lang_inputs.input_ids
#         lang_attention_mask = lang_inputs.attention_mask

#         # RGB
#         rgb = self.rgb_process(obs['rgb_obs']['rgb_static'])
#         rgb_data = rgb.unsqueeze(0)

#         # Attention mask
#         attention_mask = torch.ones(1, self.seq_len).long()
        
#         # Forward pass
#         tokenized_text = tokenized_text.to(self.device)
#         lang_attention_mask = lang_attention_mask.to(self.device) if lang_attention_mask is not None else None
#         rgb_data = rgb_data.to(self.device)
#         attention_mask = attention_mask.to(self.device)
#         rgb = rgb.to(self.device)
#         prev_actions = self.prev_action_buffer.to(self.device)

#         with torch.no_grad():
#             prediction = self.policy(
#                 rgb=rgb_data, 
#                 language=tokenized_text,
#                 attention_mask=attention_mask,
#                 prev_actions=prev_actions,
#                 train=False,
#                 lang_attention_mask=lang_attention_mask,
#                 in_simulation=True,
#             )

#         # Arm action
#         arm_action_preds = prediction['arm_action_preds']  # (1, t, chunk_size, act_dim - 1)
#         if self.pred_discrete_arm_action:
#             arm_action_preds = arm_action_preds.view(-1, self.act_dim - 1, 3)
#         else:
#             arm_action_preds = arm_action_preds.view(-1, self.act_dim - 1)  # (t*chunk_size, act_dim - 1)
        
#         # Gripper action
#         gripper_action_preds = prediction['gripper_action_preds']  # (1, t, chunk_size, 1)
#         gripper_action_preds = gripper_action_preds.view(-1, 1)  # (t*chunk_size, 1)

#         # Use the first test_chunk_size action for execution
#         arm_action_pred = arm_action_preds[:self.test_chunk_size]  # (test_chunk_size, act_dim - 1)
#         gripper_action_pred = gripper_action_preds[:self.test_chunk_size]  # (test_chunk_size, 1)
        
#         if self.is_gripper_binary:
#             gripper_action_pred = ((gripper_action_pred > 0).float()) * 2.0 - 1.0
        
#         if self.pred_discrete_arm_action:
#             arm_action_pred = arm_action_pred.softmax(dim=-1).argmax(dim=-1)
            
#         action_pred = torch.cat((arm_action_pred, gripper_action_pred), dim=-1)  # (test_chunk_size, act_dim)
#         action_pred = action_pred.detach().cpu()

#         # ============ MODIFIED: Update prev action buffer with ALL predicted actions ============
        
#         # Process ALL predicted actions (not just the first one)
#         all_arm_actions = arm_action_preds  # (total_predicted_actions, act_dim - 1)
#         all_gripper_actions = gripper_action_preds  # (total_predicted_actions, 1)
        
#         if self.is_gripper_binary:
#             all_gripper_actions = ((all_gripper_actions > 0).float()) * 2.0 - 1.0
        
#         if self.pred_discrete_arm_action:
#             all_arm_actions = all_arm_actions.softmax(dim=-1).argmax(dim=-1)
            
#         # Combine all predicted actions
#         all_predicted_actions = torch.cat((all_arm_actions, all_gripper_actions), dim=-1)  # (total_predicted_actions, act_dim)
#         all_predicted_actions = all_predicted_actions.detach().cpu()
        
#         # Update prev action buffer with ALL predicted actions
#         total_new_actions = all_predicted_actions.shape[0]
        
#         if total_new_actions >= self.prev_action_buffer_size:
#             # If we have more predictions than buffer size, take the most recent ones
#             self.prev_action_buffer = all_predicted_actions[-self.prev_action_buffer_size:].unsqueeze(0)
#         else:
#             # Shift buffer and add new actions
#             self.prev_action_buffer = torch.cat([
#                 self.prev_action_buffer[:, total_new_actions:], 
#                 all_predicted_actions.unsqueeze(0)
#             ], dim=1)
        
#         # ============ End of modification ============
        
#         self.rollout_step_counter += 1
#         return action_pred  # Still return only the actions to be executed