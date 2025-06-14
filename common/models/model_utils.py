import pyrootutils
pyrootutils.setup_root(__file__, indicator='.project-root', pythonpath=True, dotenv=True)
import omegaconf
import hydra
import os
import sys
import torch
from action_gpt.src.models.action_gpt_policy_wrapper import ActionGPT_PolicyWrapper

from transformers import AutoTokenizer
from common.processors.preprocessor_utils import get_model_vision_basic_config
import json

def load_model(pretrained_path):
    config_path = os.path.join(pretrained_path, "config.yaml")
    checkpoint_path = os.path.join(pretrained_path, "pytorch_model.bin")

    config = omegaconf.OmegaConf.load(config_path)
    model = hydra.utils.instantiate(config)
    model.config = config
    model.load_state_dict(torch.load(checkpoint_path), strict=False)

    return model

def load_action_gpt_policy(args):
    print(f"loading model from {args.action_gpt_path} ...")
    action_gpt = load_model(args.action_gpt_path)
    action_gpt_config = action_gpt.config

    lang_tokenizer = AutoTokenizer.from_pretrained(action_gpt_config['model_lang']['pretrained_model_name_or_path'])
    model_vision_basic_config = get_model_vision_basic_config(action_gpt_config['model_vision']['pretrained_model_name_or_path'])

    variant = {
        'test_chunk_size': args.test_chunk_size,
        'is_gripper_binary': args.is_gripper_binary,
        'act_dim': action_gpt_config['act_dim'],
        'seq_len': action_gpt_config['sequence_length'],
        'chunk_size': action_gpt_config['chunk_size'],
        'prev_action_buffer_size': action_gpt_config.get('prev_action_buffer_size', 10),
        'pred_discrete_arm_action': action_gpt_config.get('pred_discrete_arm_action', False),
        'is_prev_action_buffer': action_gpt_config.get('is_prev_action_buffer', True)  # 새로 추가
    }
    variant.update(model_vision_basic_config)

    policy_wrapper = ActionGPT_PolicyWrapper(
        policy=action_gpt,
        variant=variant,
        lang_tokenizer=lang_tokenizer
    )

    return policy_wrapper



# Conditionally import Moto-GPT
def load_moto_gpt_wrapper():
    try:
        # Add Moto-GPT path for import
        sys.path.append('/home/kaurml/Moto')
        # Note: The filename has a typo - it's "wraper" not "wrapper"
        from moto_gpt.src.models.moto_gpt_policy_wraper import MotoGPT_PolicyWraper
        return MotoGPT_PolicyWraper
    except ImportError as e:
        print(f"Warning: Could not import MotoGPT_PolicyWraper: {e}")
        return None

def load_moto_gpt_policy(args):
    print(f"loading Moto-GPT from {args.moto_gpt_path} ...")
    
    # Load MotoGPT_PolicyWraper dynamically (note: typo in original filename)
    MotoGPT_PolicyWraper = load_moto_gpt_wrapper()
    if MotoGPT_PolicyWraper is None:
        raise ImportError("Could not load MotoGPT_PolicyWraper. Please check Moto-GPT installation.")
    
    # Load Moto-GPT model
    def load_moto_model(pretrained_path):
        config_path = os.path.join(pretrained_path, "config.yaml")
        checkpoint_path = os.path.join(pretrained_path, "pytorch_model.bin")

        config = omegaconf.OmegaConf.load(config_path)
        
        # Moto-GPT uses 'action_chunk_size' instead of 'chunk_size'
        # Add chunk_size to config for compatibility if missing
        if 'chunk_size' not in config:
            config['chunk_size'] = config.get('action_chunk_size', 5)
        
        model = hydra.utils.instantiate(config)
        model.config = config
        
        # Load state dict with strict=False to handle potential mismatches
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
        
        return model
    
    moto_gpt = load_moto_model(args.moto_gpt_path)
    moto_gpt.mask_latent_motion_probability = args.mask_latent_motion_probability
    moto_gpt.config['mask_latent_motion_probability'] = args.mask_latent_motion_probability
    moto_gpt_config = moto_gpt.config

    lang_tokenizer = AutoTokenizer.from_pretrained(moto_gpt_config['model_lang']['pretrained_model_name_or_path'])
    model_vision_basic_config = get_model_vision_basic_config(moto_gpt_config['model_vision']['pretrained_model_name_or_path'])

    variant = {
        'test_chunk_size': args.test_chunk_size,
        'is_gripper_binary': args.is_gripper_binary,
        'use_temporal_ensemble': args.use_temporal_ensemble,

        'act_dim': moto_gpt_config['act_dim'],
        'seq_len': moto_gpt_config['sequence_length'],
        'chunk_size': moto_gpt_config.get('action_chunk_size', moto_gpt_config.get('chunk_size', 5)),
        'mask_latent_motion_probability': moto_gpt_config['mask_latent_motion_probability'],
        'latent_motion_pred': moto_gpt_config['latent_motion_pred'],
        'per_latent_motion_len': moto_gpt_config['per_latent_motion_len'],
        'pred_discrete_arm_action': moto_gpt_config.get('pred_discrete_arm_action', False)
    }
    variant.update(model_vision_basic_config)

    latent_motion_decoding_kwargs = {
        'temperature': args.temperature, 
        'sample': args.sample, 
        'top_k': args.top_k, 
        'top_p': args.top_p,
        'beam_size': args.beam_size, 
        'parallel': args.parallel
    }

    eva = MotoGPT_PolicyWraper(
        policy=moto_gpt,
        variant=variant,
        latent_motion_decoding_kwargs=latent_motion_decoding_kwargs,
        lang_tokenizer=lang_tokenizer
    )

    return eva