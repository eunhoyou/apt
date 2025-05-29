import yaml
import numpy as np
import os
import argparse
from pickle import loads
import lmdb

def check_action_bounds_from_yaml(dataset_path):
    """statistics.yaml 파일에서 action bounds 확인"""
    print("=== Action Bounds from statistics.yaml ===")
    
    for split in ['training', 'validation']:
        yaml_path = os.path.join(dataset_path, split, 'statistics.yaml')
        if os.path.exists(yaml_path):
            with open(yaml_path, 'r') as f:
                stats = yaml.safe_load(f)
            
            print(f"\n{split.capitalize()} data:")
            print(f"  act_min_bound: {stats['act_min_bound']}")
            print(f"  act_max_bound: {stats['act_max_bound']}")
            
            # 각 차원별로 보기 좋게 출력
            print("\n  Per-dimension bounds:")
            dims = ['x', 'y', 'z', 'rx', 'ry', 'rz', 'gripper']
            for i, dim in enumerate(dims):
                print(f"    {dim:8s}: [{stats['act_min_bound'][i]:8.4f}, {stats['act_max_bound'][i]:8.4f}]")

def check_action_bounds_from_lmdb(lmdb_path, num_samples=10000):
    """LMDB 데이터에서 실제 action 값의 범위 확인"""
    print("\n=== Actual Action Bounds from LMDB ===")
    
    env = lmdb.open(lmdb_path, readonly=True, create=False, lock=False)
    
    # 모든 액션 값을 저장할 리스트
    rel_actions = []
    abs_actions = []
    
    with env.begin() as txn:
        total_steps = loads(txn.get('cur_step'.encode())) + 1
        
        # 샘플링할 스텝 수 결정
        sample_steps = min(num_samples, total_steps)
        step_indices = np.random.choice(total_steps, sample_steps, replace=False)
        
        for idx in step_indices:
            # 상대 액션
            if txn.get(f'rel_action_{idx}'.encode()):
                rel_action = loads(txn.get(f'rel_action_{idx}'.encode()))
                rel_actions.append(rel_action)
            
            # 절대 액션 (있는 경우)
            if txn.get(f'abs_action_{idx}'.encode()):
                abs_action = loads(txn.get(f'abs_action_{idx}'.encode()))
                abs_actions.append(abs_action)
    
    env.close()
    
    # 상대 액션 통계
    if rel_actions:
        rel_actions = np.array(rel_actions)
        print(f"\nRelative Actions (sampled {len(rel_actions)} steps):")
        print("  Dimension |    Min    |    Max    |   Mean    |   Std")
        print("  " + "-" * 55)
        dims = ['x', 'y', 'z', 'rx', 'ry', 'rz', 'gripper']
        for i, dim in enumerate(dims):
            print(f"  {dim:8s} | {np.min(rel_actions[:, i]):9.4f} | {np.max(rel_actions[:, i]):9.4f} | "
                  f"{np.mean(rel_actions[:, i]):9.4f} | {np.std(rel_actions[:, i]):9.4f}")
    
    # 절대 액션 통계 (있는 경우)
    if abs_actions:
        abs_actions = np.array(abs_actions)
        print(f"\nAbsolute Actions (sampled {len(abs_actions)} steps):")
        print("  Dimension |    Min    |    Max    |   Mean    |   Std")
        print("  " + "-" * 55)
        for i, dim in enumerate(dims):
            print(f"  {dim:8s} | {np.min(abs_actions[:, i]):9.4f} | {np.max(abs_actions[:, i]):9.4f} | "
                  f"{np.mean(abs_actions[:, i]):9.4f} | {np.std(abs_actions[:, i]):9.4f}")

def main():
    parser = argparse.ArgumentParser(description="CALVIN 데이터셋의 action bounds 확인")
    parser.add_argument(
        "--dataset_path", 
        type=str, 
        default="/data/task_ABC_D",
        help="CALVIN 데이터셋 경로"
    )
    args = parser.parse_args()
    print(f"데이터셋 경로: {args.dataset_path}\n")
    parser.add_argument(
        "--lmdb_path", 
        type=str, 
        default="/data/lmdb_datasets/task_ABC_D/train",
        help="LMDB 데이터셋 경로"
    )
    parser.add_argument(
        "--check_lmdb", 
        action='store_true',
        help="LMDB에서 실제 액션 값 확인"
    )
    parser.add_argument(
        "--num_samples", 
        type=int, 
        default=10000,
        help="LMDB에서 샘플링할 스텝 수"
    )
    args = parser.parse_args()
    
    # statistics.yaml에서 bounds 확인
    check_action_bounds_from_yaml(args.dataset_path)
    
    # LMDB에서 실제 값 확인
    if args.check_lmdb:
        for split in ['train', 'val']:
            lmdb_split_path = os.path.join(args.lmdb_path, split)
            if os.path.exists(lmdb_split_path):
                print(f"\n\n{split.upper()} LMDB Data:")
                check_action_bounds_from_lmdb(lmdb_split_path, args.num_samples)

if __name__ == '__main__':
    main()