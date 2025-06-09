import torch
import torch.nn as nn
import torch.nn.functional as F

class ActionGPT(nn.Module):
    def __init__(
            self,
            model_lang,
            model_vision,
            model_causal_transformer,
            act_dim,
            hidden_size,
            sequence_length,
            chunk_size,
            prev_action_buffer_size,
            img_feat_dim,
            patch_feat_dim,
            lang_feat_dim,
            robot_obs_dim=15,
            robot_states_dim=7,  # tcp_pos(3) + tcp_ori(3) + gripper_action(1)
            freeze_lang=True,
            freeze_vision=True,
            pred_discrete_arm_action=False,
            is_prev_action_buffer=True,
            use_robot_obs=True,
            use_gripper_rgb=False,
            **kwargs
    ):
        super().__init__()
        self.act_dim = act_dim
        self.sequence_length = sequence_length
        self.chunk_size = chunk_size
        self.prev_action_buffer_size = prev_action_buffer_size
        self.is_prev_action_buffer = is_prev_action_buffer
        self.robot_obs_dim = robot_obs_dim
        self.robot_states_dim = robot_states_dim  # 7차원 robot states
        self.use_robot_obs = use_robot_obs
        self.use_gripper_rgb = use_gripper_rgb
        
        # GPT-2 backbone
        self.hidden_size = hidden_size
        self.model_causal_transformer = model_causal_transformer

        # Language Encoder (T5-base)
        self.model_lang = model_lang
        self.freeze_lang = freeze_lang
        if freeze_lang:
            for _, param in self.model_lang.named_parameters():
                param.requires_grad = False
        
        # Vision Encoder (SigLIP) - static RGB와 gripper RGB 공유
        self.model_vision = model_vision
        self.freeze_vision = freeze_vision
        if freeze_vision:
            for _, param in self.model_vision.named_parameters():
                param.requires_grad = False
                
        self.lang_feat_dim = lang_feat_dim
        self.img_feat_dim = img_feat_dim
        self.patch_feat_dim = patch_feat_dim
        
        # Condition embedding
        self.embed_condition = nn.Embedding(1, hidden_size)
        
        # Embedding for languages
        self.embed_lang = torch.nn.Linear(self.lang_feat_dim, hidden_size)
        
        # Vision MLP projectors (ViT 이후 MLP로 처리)
        self.embed_static_img = torch.nn.Linear(self.img_feat_dim, hidden_size)
        self.embed_static_patch = torch.nn.Linear(self.patch_feat_dim, hidden_size)
        
        if self.use_gripper_rgb:
            self.embed_gripper_img = torch.nn.Linear(self.img_feat_dim, hidden_size)
            self.embed_gripper_patch = torch.nn.Linear(self.patch_feat_dim, hidden_size)
        
        # Robot states embedding (action encoder와 유사한 MLP 구조)
        if self.use_robot_obs:
            self.embed_robot_states = nn.Sequential(
                nn.Linear(self.robot_states_dim, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, hidden_size)
            )
        
        # Embedding functions for previous actions (action encoder와 유사한 구조)
        if self.is_prev_action_buffer:
            self.embed_prev_action = nn.Sequential(
                nn.Linear(act_dim, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, hidden_size)
            )
        
        # Action query tokens
        self.action_queries = nn.Embedding(1, hidden_size)
        self.action_chunk_queries = nn.Embedding(sequence_length*chunk_size, hidden_size)
        self.action_chunk_queries.weight.data.fill_(0)
        
        # Layer norm
        self.embed_ln = nn.LayerNorm(hidden_size)

        # Cross-attention for vision features
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            batch_first=True,
            dropout=0.1
        )

        # Action prediction head
        self.pred_act_mlps = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2, hidden_size//2)
        ])
        
        self.pred_discrete_arm_action = pred_discrete_arm_action
        if self.pred_discrete_arm_action:
            self.pred_arm_act = nn.Linear(hidden_size//2, (self.act_dim-1)*3)
        else:
            self.pred_arm_act = nn.Linear(hidden_size//2, self.act_dim-1)

        self.pred_gripper_act = nn.Linear(hidden_size//2, 1)
        
    @property
    def device(self):
        return next(self.parameters()).device

    def get_state_dict_to_save(self):
        modules_to_exclude = ['model_lang', 'model_vision']
        state_dict = {k: v for k, v in self.state_dict().items() if
                      not any(module_name in k for module_name in modules_to_exclude)}
        return state_dict

    def extract_robot_states(self, robot_obs):
        """
        Extract 7-dim robot states from 15-dim robot_obs
        robot_obs: tcp_pos(3) + tcp_ori(3) + gripper_width(1) + arm_joints(7) + gripper_action(1)
        Extract: tcp_pos(3) + tcp_ori(3) + gripper_action(1) = 7 dims
        """
        tcp_pos = robot_obs[..., :3]           # (batch_size, 3)
        tcp_ori = robot_obs[..., 3:6]          # (batch_size, 3)
        gripper_action = robot_obs[..., -1:]   # (batch_size, 1)
        
        robot_states = torch.cat([tcp_pos, tcp_ori, gripper_action], dim=-1)  # (batch_size, 7)
        return robot_states

    def create_attention_mask(self, batch_size, n_lang_tokens, n_robot_states_tokens, n_prev_action_tokens, n_action_query_tokens, device):
        """
        Create attention mask:
        
        Input tokens (Language, Robot states, Prev actions):
        - Fully bidirectional among themselves 
        - CANNOT attend to action queries (future actions)
        
        Action queries:
        - Can attend to all input tokens
        - Bidirectional among themselves
        
        Attention Pattern:
        Lang ↔ Robot states ↔ Prev actions (bidirectional)
        Action queries → All inputs + Action queries ↔ Action queries
        """
        total_length = n_lang_tokens + n_robot_states_tokens + n_prev_action_tokens + n_action_query_tokens
        mask = torch.zeros((batch_size, 1, total_length, total_length), device=device)
        
        # Calculate boundary indices
        lang_end = n_lang_tokens
        robot_end = lang_end + n_robot_states_tokens  
        prev_action_end = robot_end + n_prev_action_tokens
        action_query_start = prev_action_end
        
        # 1. Input tokens (Language + Robot states + Prev actions) - Fully bidirectional among themselves
        input_end = prev_action_end
        mask[:, :, :input_end, :input_end] = 1
        
        # 2. Action queries can attend to all input tokens
        mask[:, :, action_query_start:, :input_end] = 1
        
        # 3. Action queries are bidirectional among themselves  
        mask[:, :, action_query_start:, action_query_start:] = 1
        
        # 4. Input tokens CANNOT attend to action queries (no future information leakage)
        # mask[:, :, :input_end, action_query_start:] = 0  # 이미 0으로 초기화됨
        
        return mask

    def forward(self, 
                rgb_static,           # (b, 1, c, h, w) static RGB
                language,             # Tokenized language input
                rgb_gripper=None,     # (b, 1, c, h, w) gripper RGB
                robot_obs=None,       # (b, robot_obs_dim) robot observation
                prev_actions=None,    # (b, prev_action_buffer_size, act_dim)
                lang_attention_mask=None,
                **kwargs
    ):
        batch_size, _, c, h, w = rgb_static.shape
        
        # 1. Language embedding with T5-base
        if self.freeze_lang:
            with torch.no_grad():
                lang_embeddings = self.model_lang(input_ids=language, attention_mask=lang_attention_mask).last_hidden_state
        else:
            lang_embeddings = self.model_lang(input_ids=language, attention_mask=lang_attention_mask).last_hidden_state
        lang_embeddings = self.embed_lang(lang_embeddings.float())  # (b, n_lang_tokens, h)

        # 2. Static vision features (ViT + MLP projector)
        if self.freeze_vision:
            with torch.no_grad():
                static_obs_embeddings, static_patch_embeddings = self.model_vision(rgb_static.reshape(batch_size, c, h, w))
        else:
            static_obs_embeddings, static_patch_embeddings = self.model_vision(rgb_static.reshape(batch_size, c, h, w))
        
        # MLP projection for static vision
        static_obs_embeddings = self.embed_static_img(static_obs_embeddings.float())  # (b, 1, h)
        static_patch_embeddings = self.embed_static_patch(static_patch_embeddings.float())  # (b, n_patches, h)
        
        # 3. Gripper vision features (ViT + MLP projector)
        if self.use_gripper_rgb and rgb_gripper is not None:
            if self.freeze_vision:
                with torch.no_grad():
                    gripper_obs_embeddings, gripper_patch_embeddings = self.model_vision(rgb_gripper.reshape(batch_size, c, h, w))
            else:
                gripper_obs_embeddings, gripper_patch_embeddings = self.model_vision(rgb_gripper.reshape(batch_size, c, h, w))
            
            # MLP projection for gripper vision
            gripper_obs_embeddings = self.embed_gripper_img(gripper_obs_embeddings.float())  # (b, 1, h)
            gripper_patch_embeddings = self.embed_gripper_patch(gripper_patch_embeddings.float())  # (b, n_patches, h)
            
            # Combine static and gripper patch embeddings for cross-attention
            combined_patch_embeddings = torch.cat([static_patch_embeddings, gripper_patch_embeddings], dim=1)  # (b, 2*n_patches, h)
        else:
            combined_patch_embeddings = static_patch_embeddings  # (b, n_patches, h)
       
        # 4. Add conditional embeddings
        condition_embeddings = self.embed_condition.weight.view(1, 1, self.hidden_size)  # (1, 1, h)
        lang_embeddings = lang_embeddings.view(batch_size, -1, self.hidden_size) + condition_embeddings
        combined_patch_embeddings = combined_patch_embeddings + condition_embeddings
        
        # 5. Robot states embedding (7차원 robot states + MLP)
        if self.use_robot_obs and robot_obs is not None:
            robot_states = self.extract_robot_states(robot_obs)  # (b, 7)
            robot_states_embeddings = self.embed_robot_states(robot_states.float())  # (b, h)
            robot_states_embeddings = robot_states_embeddings.unsqueeze(1) + condition_embeddings  # (b, 1, h)
        else:
            robot_states_embeddings = torch.zeros(batch_size, 0, self.hidden_size, device=rgb_static.device)
        
        # 6. Previous actions embedding (action encoder와 유사한 MLP)
        if self.is_prev_action_buffer:
            if prev_actions is not None:
                # Reshape to process all actions through MLP
                b, seq_len, act_dim = prev_actions.shape
                prev_actions_flat = prev_actions.view(-1, act_dim)  # (b*seq_len, act_dim)
                prev_action_embeddings_flat = self.embed_prev_action(prev_actions_flat.float())  # (b*seq_len, h)
                prev_action_embeddings = prev_action_embeddings_flat.view(b, seq_len, self.hidden_size)  # (b, seq_len, h)
                prev_action_embeddings = prev_action_embeddings + condition_embeddings
            else:
                prev_action_embeddings = torch.zeros(batch_size, self.prev_action_buffer_size, self.hidden_size, device=rgb_static.device)
        else:
            prev_action_embeddings = torch.zeros(batch_size, 0, self.hidden_size, device=rgb_static.device)
        
        # 7. Generate action query tokens
        action_chunk_queries = self.action_queries.weight + self.action_chunk_queries.weight
        action_chunk_queries = action_chunk_queries.view(1, self.sequence_length*self.chunk_size, self.hidden_size)
        action_chunk_queries = action_chunk_queries.expand(batch_size, -1, -1)
        
        # 8. Concatenate transformer input sequence: [language, robot_states, prev_actions, action_queries]
        n_lang_tokens = lang_embeddings.shape[1]
        n_robot_states_tokens = robot_states_embeddings.shape[1]
        n_prev_action_tokens = prev_action_embeddings.shape[1]
        n_action_query_tokens = action_chunk_queries.shape[1]
        
        stacked_inputs = torch.cat([
            lang_embeddings,             # (b, n_lang_tokens, h)
            robot_states_embeddings,     # (b, n_robot_states_tokens, h) - 0 or 1
            prev_action_embeddings,      # (b, n_prev_action_tokens, h) - 0 or prev_action_buffer_size
            action_chunk_queries         # (b, n_action_query_tokens, h)
        ], dim=1)
        
        # 9. Create attention mask
        full_attention_mask = self.create_attention_mask(
            batch_size=batch_size,
            n_lang_tokens=n_lang_tokens,
            n_robot_states_tokens=n_robot_states_tokens,
            n_prev_action_tokens=n_prev_action_tokens, 
            n_action_query_tokens=n_action_query_tokens,
            device=rgb_static.device
        )
        
        stacked_inputs = self.embed_ln(stacked_inputs)
        
        # Apply language attention mask if provided
        if lang_attention_mask is not None:
            lang_mask = lang_attention_mask.view(batch_size, 1, 1, -1)
            full_attention_mask[:, :, :, :n_lang_tokens] = full_attention_mask[:, :, :, :n_lang_tokens] * lang_mask

        # trajectory_gpt2.py expects attention_mask to be converted to the right format
        # Looking at the code: if attention_mask.dim() != 4, it reshapes to (batch_size, 1, 1, seq_len)
        # So we need to provide either (batch_size, seq_len) or (batch_size, seq_len, seq_len)
        
        # Option 1: Provide 2D mask (batch_size, seq_len) - but this is for padding, not causal
        # Option 2: Provide 4D mask directly in the format GPT-2 expects
        
        # Let's provide the 4D mask directly to avoid reshaping issues
        attention_mask_for_gpt = full_attention_mask  # Keep (batch_size, 1, seq_len, seq_len)
        
        # Ensure it's the right dtype
        attention_mask_for_gpt = attention_mask_for_gpt.float()
        
        # GPT forward pass without custom attention mask for now
        # TODO: Implement proper attention mask handling
        transformer_outputs = self.model_causal_transformer(
            inputs_embeds=stacked_inputs,
            # attention_mask=attention_mask_for_gpt,  # Temporarily disabled
        )
        
        # Get hidden states
        hidden_states = transformer_outputs['last_hidden_state']
        
        # 11. Extract action query hidden states
        action_query_start_idx = n_lang_tokens + n_robot_states_tokens + n_prev_action_tokens
        action_query_end_idx = action_query_start_idx + n_action_query_tokens
        action_query_hidden = hidden_states[:, action_query_start_idx:action_query_end_idx]
        
        # 12. Cross attention with vision features
        attended_img_features, _ = self.cross_attention(
            query=action_query_hidden,
            key=combined_patch_embeddings,  # static + gripper patch embeddings
            value=combined_patch_embeddings
        )
        
        # Combine with original query (residual connection)
        action_embedding = attended_img_features + action_query_hidden
        
        # 13. Action prediction
        for i, pred_act_mlp in enumerate(self.pred_act_mlps):
            action_embedding = pred_act_mlp(action_embedding)
            if i == 0:  # ReLU after first layer
                action_embedding = F.relu(action_embedding)
        
        # Predict arm and gripper actions
        if self.pred_discrete_arm_action:
            arm_action_preds = self.pred_arm_act(action_embedding)
            arm_action_preds = arm_action_preds.reshape(batch_size, self.sequence_length, self.chunk_size, self.act_dim-1, 3)
        else:
            arm_action_preds = self.pred_arm_act(action_embedding)
            arm_action_preds = arm_action_preds.reshape(batch_size, self.sequence_length, self.chunk_size, self.act_dim-1)
            
        gripper_action_preds = self.pred_gripper_act(action_embedding)
        gripper_action_preds = gripper_action_preds.reshape(batch_size, self.sequence_length, self.chunk_size, 1)
            
        prediction = {
            'arm_action_preds': arm_action_preds,
            'gripper_action_preds': gripper_action_preds
        }
            
        return prediction