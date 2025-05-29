import numpy as np
import os
import argparse
from glob import glob

def count_calvin_episodes(dataset_path):
    """CALVIN 데이터셋의 episode 수를 확인하는 함수"""
    
    # 훈련/검증 데이터 경로
    train_path = os.path.join(dataset_path, 'training')
    val_path = os.path.join(dataset_path, 'validation')
    
    results = {}
    
    # 훈련 데이터 확인
    if os.path.exists(train_path):
        # 언어 주석 파일 로드
        lang_ann_path = os.path.join(train_path, 'lang_annotations/auto_lang_ann.npy')
        if os.path.exists(lang_ann_path):
            annotations = np.load(lang_ann_path, allow_pickle=True).item()
            train_episodes = len(annotations['info']['indx'])
            results['training'] = train_episodes
            
            # npz 파일 수 확인 (참고용)
            npz_files = glob(os.path.join(train_path, '*.npz'))
            results['training_frames'] = len(npz_files)
        else:
            print(f"Warning: {lang_ann_path} not found")
    
    # 검증 데이터 확인
    if os.path.exists(val_path):
        # 언어 주석 파일 로드
        lang_ann_path = os.path.join(val_path, 'lang_annotations/auto_lang_ann.npy')
        if os.path.exists(lang_ann_path):
            annotations = np.load(lang_ann_path, allow_pickle=True).item()
            val_episodes = len(annotations['info']['indx'])
            results['validation'] = val_episodes
            
            # npz 파일 수 확인 (참고용)
            npz_files = glob(os.path.join(val_path, '*.npz'))
            results['validation_frames'] = len(npz_files)
        else:
            print(f"Warning: {lang_ann_path} not found")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="CALVIN 데이터셋의 episode 수 확인")
    parser.add_argument(
        "--dataset_path", 
        type=str, 
        default="/data/task_ABC_D"
    )
    args = parser.parse_args()
    
    print(f"데이터셋 경로: {args.dataset_path}\n")
    
    # Episode 수 확인
    results = count_calvin_episodes(args.dataset_path)
    
    # 결과 출력
    print("=== CALVIN ABC_D Dataset Statistics ===")
    if 'training' in results:
        print(f"훈련 데이터:")
        print(f"  - Episodes: {results['training']:,}")
        print(f"  - Total frames: {results['training_frames']:,}")
        print(f"  - Avg frames/episode: {results['training_frames']/results['training']:.1f}")
    
    if 'validation' in results:
        print(f"\n검증 데이터:")
        print(f"  - Episodes: {results['validation']:,}")
        print(f"  - Total frames: {results['validation_frames']:,}")
        print(f"  - Avg frames/episode: {results['validation_frames']/results['validation']:.1f}")
    
    if results:
        total_episodes = results.get('training', 0) + results.get('validation', 0)
        total_frames = results.get('training_frames', 0) + results.get('validation_frames', 0)
        print(f"\n전체 데이터:")
        print(f"  - Total episodes: {total_episodes:,}")
        print(f"  - Total frames: {total_frames:,}")

if __name__ == '__main__':
    main()