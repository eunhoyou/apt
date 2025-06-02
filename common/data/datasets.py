import lmdb
from pickle import loads
import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset
from torchvision.io import decode_jpeg
import cv2
import os
from einops import rearrange
import random
import json
from PIL import Image
import random

def get_split_and_ratio(split, splits):
    assert split in ['train', 'val']
    assert 'train' in splits
    if 'val' in splits:
        start_ratio=0
        end_ratio=1
    else:
        if split == 'train':
            start_ratio=0
            end_ratio=0.95
        else:
            split = 'train' 
            start_ratio=0.95
            end_ratio=1
    return split, start_ratio, end_ratio

class DataPrefetcher():
    def __init__(self, loader, device, lang_tokenizer=None):
        self.device = device
        self.loader = loader
        self.lang_tokenizer = lang_tokenizer
        self.iter = iter(self.loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.batch = next(self.iter)
        except StopIteration:
            self.batch = None
            self.iter = iter(self.loader)
            return 
        if self.lang_tokenizer is not None:
            lang_inputs = self.lang_tokenizer(self.batch['lang'], return_tensors="pt", padding=True)
            lang_input_ids = lang_inputs.input_ids
            lang_attention_mask = lang_inputs.attention_mask
            self.batch["lang_input_ids"] = lang_input_ids
            self.batch["lang_attention_mask"] = lang_attention_mask

        with torch.cuda.stream(self.stream):
            for key in self.batch:
                if type(self.batch[key]) is torch.Tensor:
                    self.batch[key] = self.batch[key].to(self.device, non_blocking=True)

    def __len__(self):
        return len(self.loader.dataset)

    def next(self):
        batch = self.batch
        if batch is not None:
            for key in batch:
                if type(batch[key]) is torch.Tensor:
                    batch[key].record_stream(torch.cuda.current_stream())
        self.preload()
        return batch

    def next_without_none(self):
        batch = self.next()
        if batch is None:
            batch = self.next()
        return batch


class LMDBDataset_for_ActionGPT(Dataset):
    def __init__(
        self, lmdb_dir, split,
        sequence_length,
        chunk_size,
        prev_action_buffer_size,
        action_dim=7,
        video_dir=None, rgb_shape=(224, 224), rgb_preprocessor=None, max_skip_frame=None):

        super().__init__()

        self.sequence_length = sequence_length
        self.chunk_size = chunk_size
        self.prev_action_buffer_size = prev_action_buffer_size
        self.action_dim = action_dim

        self.dummy_rgb_initial = torch.zeros(1, 3, rgb_shape[0], rgb_shape[1], dtype=torch.uint8)
        self.dummy_actions = torch.zeros(sequence_length, chunk_size, action_dim)
        self.dummy_mask = torch.zeros(sequence_length, chunk_size)
        self.dummy_prev_actions = torch.zeros(prev_action_buffer_size, action_dim)

        self.lmdb_dir = lmdb_dir
        self.video_dir = video_dir
        self.rgb_preprocessor = rgb_preprocessor

        split, start_ratio, end_ratio = get_split_and_ratio(split, os.listdir(lmdb_dir))
        self.split = split
        env = lmdb.open(os.path.join(lmdb_dir, split), readonly=True, create=False, lock=False)
        with env.begin() as txn:
            dataset_len = loads(txn.get('cur_step'.encode())) + 1
            self.start_step = int(dataset_len * start_ratio) 
            self.end_step = int(dataset_len * end_ratio) - sequence_length - chunk_size
        env.close()

    def open_lmdb(self):
        self.env = lmdb.open(os.path.join(self.lmdb_dir, self.split), readonly=True, create=False, lock=False)
        self.txn = self.env.begin()

    def extract_lang_goal(self, idx, cur_episode):
        feature_dict = loads(self.txn.get(f'feature_dict_{idx}'.encode()))
        lang = feature_dict['observation']['natural_language_instruction'].decode().lower().strip('.')
        return lang

    def get_video_path(self, cur_episode):
        # return os.path.join(self.video_dir, f'{self.split}_eps_{cur_episode:08d}.mp4')
        raise NotImplementedError

    def extract_frames(self, idx, cur_episode, rgb_initial):
        start_local_step = loads(self.txn.get(f'local_step_{idx}'.encode()))
        video_path = self.get_video_path(cur_episode)
        video = cv2.VideoCapture(video_path)

        def _extract_frame(frame_idx):
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = video.read()
            try:
                assert ret is True
            except Exception as e:
                # print(f"Failed to read video (path={video_path}, frame_idx={frame_idx})")
                raise e
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = torch.from_numpy(rearrange(frame, 'h w c -> c h w'))
            
            if self.rgb_preprocessor is not None:
                frame  = self.rgb_preprocessor(frame)
            return frame

        rgb_initial[0] = _extract_frame(start_local_step)

        video.release()

    def extract_actions(self, idx, cur_episode, actions, mask):
        for i in range(self.sequence_length):
            for j in range(self.chunk_size):
                cur_idx = idx + j
                if loads(self.txn.get(f'cur_episode_{cur_idx}'.encode())) == cur_episode:
                    mask[i, j] = 1
                    action = self.extract_action(cur_idx)
                    actions[i, j] = action

    def extract_action(self, idx):
        raise NotImplementedError

    def __getitem__(self, idx):
        if hasattr(self, 'env') == 0:
            self.open_lmdb()
            
        orig_idx = idx
        idx = idx + self.start_step
        cur_episode = loads(self.txn.get(f'cur_episode_{idx}'.encode()))

        rgb_initial = self.dummy_rgb_initial.clone()
        actions = self.dummy_actions.clone()
        mask = self.dummy_mask.clone()
        prev_actions = self.dummy_prev_actions.clone()
        prev_actions_mask = torch.zeros(self.prev_action_buffer_size)
        
        # previous actions 
        for i in range(1, self.prev_action_buffer_size + 1):
            prev_idx = idx - i
            if prev_idx >= self.start_step and loads(self.txn.get(f'cur_episode_{prev_idx}'.encode())) == cur_episode:
                prev_actions[self.prev_action_buffer_size - i] = self.extract_action(prev_idx)
                prev_actions_mask[self.prev_action_buffer_size - i] = 1.0

        # extract lang goal
        lang = self.extract_lang_goal(idx, cur_episode)

        # extract initial frame
        self.extract_frames(
            idx=idx, 
            cur_episode=cur_episode, 
            rgb_initial=rgb_initial
        )

        self.extract_actions(
            idx=idx, 
            cur_episode=cur_episode,
            actions=actions, 
            mask=mask
        )

        return {
            "lang": lang,
            "rgb_initial": rgb_initial,
            "actions": actions,
            "mask": mask,
            "prev_actions": prev_actions,
            "prev_actions_mask": prev_actions_mask,
            "idx": orig_idx
        }

    def __len__(self):
        return self.end_step - self.start_step

class LMDBDataset_for_ActionGPT_CALVIN(LMDBDataset_for_ActionGPT):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def extract_lang_goal(self, idx, cur_episode):
        lang = loads(self.txn.get(f'inst_{cur_episode}'.encode()))
        return lang

    def extract_frames(self, idx, cur_episode, rgb_initial):
        rgb_initial[0] = decode_jpeg(loads(self.txn.get(f'rgb_static_{idx}'.encode())))

    def extract_actions(self, idx, cur_episode, actions, mask):
        for i in range(self.sequence_length):
            for j in range(self.chunk_size):
                cur_idx = idx + j
                if loads(self.txn.get(f'cur_episode_{cur_idx}'.encode())) == cur_episode:
                    mask[i, j] = 1
                    action = self.extract_action(cur_idx)
                    actions[i, j] = action

    def extract_action(self, idx):
        action = loads(self.txn.get(f'rel_action_{idx}'.encode()))
        action[-1] = (action[-1] + 1) / 2
        return action