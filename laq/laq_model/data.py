from functools import partial
import json
from pathlib import Path
from typing import List
from PIL import Image

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader as PytorchDataLoader

from torchvision import transforms as T

import os
import random


def exists(val):
    return val is not None

def identity(t, *args, **kwargs):
    return t

def pair(val):
    return val if isinstance(val, tuple) else (val, val)

'''
This is the dataset class for Sthv2 dataset.
The dataset is a list of folders, each folder contains a sequence of frames.
You have to change the dataset class to fit your dataset for custom training.
'''

class ImageVideoDataset(Dataset):
    def __init__(
        self,
        folder,
        image_size,
        offset=5,
    ):
        super().__init__()
        
        self.folder = folder
        self.folder_list = os.listdir(folder)
        self.image_size = image_size
      
        self.offset = offset

        self.transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize(image_size),
            T.ToTensor(),
        ])


    def __len__(self):
        return len(self.folder_list) ## length of folder list is not exact number of frames; TODO: change this to actual number of frames
    
    def __getitem__(self, index):
        try :
            offset = self.offset
            
            folder = self.folder_list[index]
            img_list = os.listdir(os.path.join(self.folder, folder))

            img_list = sorted(img_list, key=lambda x: int(x.split('.')[0][4:]))
            ## pick random frame 
            first_frame_idx = random.randint(0, len(img_list)-1)
            first_frame_idx = min(first_frame_idx, len(img_list)-1)
            second_frame_idx = min(first_frame_idx + offset, len(img_list)-1)
            
            first_path = os.path.join(self.folder, folder, img_list[first_frame_idx])
            second_path = os.path.join(self.folder, folder, img_list[second_frame_idx])
                    
            img = Image.open(first_path)
            next_img = Image.open(second_path)
            
            transform_img = self.transform(img).unsqueeze(1)
            next_transform_img = self.transform(next_img).unsqueeze(1)
            
            cat_img = torch.cat([transform_img, next_transform_img], dim=1)
            return cat_img
        except :
            print("error", index)
            if index < self.__len__() - 1:
                return self.__getitem__(index + 1)
            else:
                return self.__getitem__(random.randint(0, self.__len__() - 1))


def video_to_tensor(
    path: str,              # Path of the video to be imported
    offset: int = 1,        # Number of frames to skip after first
    transform = T.ToTensor(),       # Transform to be applied to each frame
) -> torch.Tensor:          # shape (1, channels, frames, height, width)

    video = cv2.VideoCapture(path)
    # frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    ## pick random frame 
    # first_frame_idx = random.randint(0, frame_count - 1)
    # first_frame_idx = min(first_frame_idx, frame_count - 1)
    # second_frame_idx = min(first_frame_idx + offset, frame_count - 1)
    # frame_indices = [first_frame_idx, second_frame_idx]

    frames = []
    check = True

    # for fidx in frame_indices:
    #     video.set(cv2.CAP_PROP_POS_FRAMES, fidx)
    #     check, frame = video.read()
    #     if not check:
    #         print(path, frame_indices, frame_count)
    #         break
    #     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #     frames.append(Image.fromarray(frame))
    while check:
        check, frame = video.read()
        if not check:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame))

    frame_count = len(frames)
    first_frame_idx = random.randint(0, frame_count - 1)
    first_frame_idx = min(first_frame_idx, frame_count - 1)
    second_frame_idx = min(first_frame_idx + offset, frame_count - 1)
    frame_indices = [first_frame_idx, second_frame_idx]
    frames_sampled = [frames[i] for i in frame_indices]
    frames_torch = tuple(map(transform, frames_sampled))
    frames_torch = torch.stack(frames_torch, dim = 1)

    return frames_torch


# video dataset
def process_ssv2_videos(root_dir: Path, mode: str) -> List[Path]:
    if mode == 'train':
        label_paths = [root_dir / 'labels' / 'train.json']
    elif mode == 'val':
        label_paths = [root_dir / 'labels' / 'validation.json']
    elif mode == 'trainval':
        label_paths = [
            root_dir / 'labels' / 'train.json',
            root_dir / 'labels' / 'validation.json'
        ]
    elif mode == 'test':
        label_paths = [root_dir / 'labels' / 'test.json']
    elif mode == 'all':
        label_paths = [
            root_dir / 'labels' / 'train.json',
            root_dir / 'labels' / 'validation.json',
            root_dir / 'labels' / 'test.json'
        ]

    paths: List[Path] = []
    print(root_dir)
    for label_path in label_paths:
        with open(label_path, 'r') as f:
            data = json.load(f)
        for ent in data:
            vid_name = ent['id'] + '.webm'
            paths.append(root_dir / '20bn-something-something-v2' / vid_name)
    return paths


class VideoDataset(Dataset):
    def __init__(
        self,
        folder: List[str],
        image_size: int,
        mode: str = 'train',
        offset: int = 5,
    ):
        super().__init__()
        
        self.folder = folder
        self.image_size = (image_size, image_size) if isinstance(image_size, int) else image_size     
        self.offset = offset
        self.video_list = []
        for data_dir in folder:
            if "something-something-v2" in data_dir:
                self.video_list.extend(process_ssv2_videos(Path(f'{data_dir}'), mode))
            if "ego4d" in folder:
                raise NotImplementedError("ego4d dataset not implemented yet")        

        self.transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize(self.image_size),
            T.ToTensor(),
        ])

        # functions to transform video path to tensor
        self.video_to_tensor = partial(video_to_tensor, offset=offset, transform = self.transform)

        print(f"Found {len(self.video_list)} videos in {mode} mode!")


    def __len__(self):
        return len(self.video_list) ## length of folder list is not exact number of frames; TODO: change this to actual number of frames
    
    def __getitem__(self, index):
        try :
            video_path = self.video_list[index]
            video_tensor = self.video_to_tensor(video_path)
            return video_tensor
        except :
            print("error", index)
            if index < self.__len__() - 1:
                return self.__getitem__(index + 1)
            else:
                return self.__getitem__(random.randint(0, self.__len__() - 1))


if __name__ == "__main__":
    # test video dataset
    dataset = VideoDataset(['/grogu/user/sroutra2/datasets/something-something-v2'], 256, mode='all')
    t1 = dataset[337]
    import ipdb; ipdb.set_trace()
    # for i in range(len(dataset)):
    #     t = dataset[i]
    