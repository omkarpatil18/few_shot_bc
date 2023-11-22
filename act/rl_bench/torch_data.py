import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import glob
import os
import random
import re
from functools import partial

CONFIG_DIM = 7  # joint space


class ConfigMinMaxNormalization(object):
    """MinMax Normalization --> [-1, 1]
    x = (x - min) / (max - min).
    x = x * 2 - 1
    """

    def __init__(self, bounds):
        assert bounds.shape[1] == 2
        self._min = bounds[:, 0]
        self._max = bounds[:, 1]

    def validate_bounds(self, js):
        return np.all(js > self._min) and np.all(js < self._max)

    def transform(self, X):
        X = 1.0 * (X - self._min) / (self._max - self._min)
        X = X * 2.0 - 1.0
        return X

    def inverse_transform(self, X):
        X = (X + 1.0) / 2.0
        X = 1.0 * X * (self._max - self._min) + self._min
        return X

    def clamp(self, js):
        min_ = np.ones(js.shape) * self._min
        max_ = np.ones(js.shape) * self._max
        js = np.minimum(np.maximum(js, min_), max_)
        return js


class ReverseTrajDataset(Dataset):
    """
    Dataset for robot trajectory for BC.
    """

    task_filter_map = {
        "box_open": re.compile(r"forward_[\d]*1.pickle"),
        "box_close": re.compile(r"backward_[\d]*1.pickle"),
        "box": re.compile(r"[a-zA-Z_]*[\d]*1.pickle"),
        "door_open": re.compile(r"forward_[\d]*2.pickle"),
        "door_close": re.compile(r"backward_[\d]*2.pickle"),
        "door": re.compile(r"[a-zA-Z_]*[\d]*2.pickle"),
        "toilet_seat_up": re.compile(r"forward_[\d]*3.pickle"),
        "toilet_seat_down": re.compile(r"backward_[\d]*3.pickle"),
        "toilet_seat": re.compile(r"[a-zA-Z_]*[\d]*3.pickle"),
    }
    # This order needs to be consistent with what you pass while training
    camera_names = [
        "front_rgb",
        "left_shoulder_rgb",
        "right_shoulder_rgb",
        "wrist_rgb",
    ]

    def __init__(
        self,
        file_list,
        required_data_keys,
        chunk_size=100,
        norm_bound=None,
        add_task_ind=False,
        sampler=partial(np.random.uniform, 0, 1),  # partial function
    ):
        self.file_list = file_list
        self.required_data_keys = required_data_keys
        self.chunk_size = chunk_size
        if norm_bound is not None:
            self.normalizer = ConfigMinMaxNormalization(norm_bound)
        else:
            self.normalizer = None
        self.len = len(self.file_list)
        self.sampler = sampler
        self.add_task_ind = add_task_ind

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        with open(self.file_list[index], "rb") as f:
            demo = pickle.load(f)
            valid_keys = []
            for key in demo:
                if len(demo[key]) > 0:
                    valid_keys.append(key)

            assert len(valid_keys) > 0
            assert all([req_key in valid_keys for req_key in self.required_data_keys])

            episode_len = len(demo[valid_keys[0]])
            # start_ts = np.random.choice(episode_len)
            # Sample start_ts from a distribution
            start_ts = int(self.sampler() * episode_len)
            end_ts = min(episode_len, start_ts + self.chunk_size)

            image_dict = {}
            data_batch = {}
            for key in self.required_data_keys:
                if "position" in key:
                    chunk_data = demo[key][start_ts:end_ts]
                    if self.normalizer is not None:
                        for i, js in enumerate(chunk_data):
                            if self.normalizer.validate_bounds(js):
                                chunk_data[i] = self.normalizer.transform(js)
                            else:
                                chunk_data[i] = self.normalizer.clamp(js)

                    data = torch.zeros((self.chunk_size, CONFIG_DIM))
                    data[: end_ts - start_ts, :] = torch.as_tensor(np.array(chunk_data))
                    data_batch["joint_action"] = data
                elif "gripper" in key:
                    chunk_data = demo[key][start_ts:end_ts]
                    data = torch.zeros((self.chunk_size))
                    data[: end_ts - start_ts] = torch.as_tensor(np.array(chunk_data))
                    data_batch["gripper_action"] = data
                elif "rgb" in key:
                    image_dict[key] = demo[key][start_ts]
                else:
                    data = demo[key][start_ts]
                    data = torch.as_tensor(data)
                    data_batch[key] = data

            all_cam_images = []
            for cam_name in ReverseTrajDataset.camera_names:
                all_cam_images.append(image_dict[cam_name])
            all_cam_images = torch.from_numpy(np.stack(all_cam_images, axis=0))
            image_data = torch.einsum(
                "k h w c -> k c h w", all_cam_images
            )  # channel last
            # normalize image and change dtype to float
            image_data = image_data / 255.0
            data_batch["images"] = image_data
            is_pad = np.ones((self.chunk_size))
            is_pad[: end_ts - start_ts] = 0
            data_batch["is_pad"] = torch.from_numpy(is_pad).bool()
            if self.add_task_ind:
                if "backward" in self.file_list[index]:
                    data_batch["task_ind"] = torch.tensor(-1)
                elif "forward" in self.file_list[index]:
                    data_batch["task_ind"] = torch.tensor(1)
            else:
                data_batch["task_ind"] = torch.tensor(0)  # no latent control

        assert data_batch["images"].shape == torch.Size([4, 3, 128, 128])
        assert data_batch["is_pad"].shape == torch.Size([self.chunk_size])
        assert data_batch["joint_action"].shape == torch.Size([self.chunk_size, 7])
        assert data_batch["gripper_action"].shape == torch.Size([self.chunk_size])
        assert data_batch["task_ind"].shape == torch.Size([])

        return data_batch


def load_data(
    dataset_dir,
    task_filter_key=ReverseTrajDataset.task_filter_map["box_open"],
    required_data_keys=[
        "front_rgb",
        "left_shoulder_rgb",
        "right_shoulder_rgb",
        "wrist_rgb",
        "joint_positions",
        "gripper_open",
    ],
    chunk_size=100,
    norm_bound=None,
    batch_size=8,
    train_split=0.8,
    add_task_ind=False,
):
    """
    Method to return a Dataloader of manipulator demonstrations
    Parameters
    ---------
    dataset_dir
        Location where the demonstrations are saved
    task_filter_keys: re.compile object
        Regex to extract specific tasks from the recorded demonstrations-
        use ReverseTrajDataset.task_filter_map
    required_data_keys: default=["front_rgb", "left_shoulder_rgb", "right_shoulder_rgb",
        "wrist_rgb", "joint_positions", "gripper_open"]
        Features to extract for an episode
    chunk_size: default=100,
        Chunk of action size to return
    norm_bound: default==None,
        Bounds for normalizing the data
    batch_size: int; default=8
        Size of the batch
    train_split: float; default=0.8
        Train-test split

    Notes
    -----
    Possible set of keys in the recorded demonstrations
    [
        'front_rgb', 'front_mask', 'front_depth', 'front_point_cloud',
        'left_shoulder_rgb', 'left_shoulder_mask', 'left_shoulder_depth',
        'left_shoulder_point_cloud', 'right_shoulder_rgb', 'right_shoulder_mask',
        'right_shoulder_depth', 'right_shoulder_point_cloud', 'overhead_rgb',
        'overhead_mask', 'overhead_depth', 'overhead_point_cloud', 'wrist_rgb',
        'wrist_mask', 'wrist_depth', 'wrist_point_cloud', 'joint_positions',
        'joint_velocities', 'gripper_pose', 'gripper_open'
    ]
    Keys returned by the dataloaders are:
        [images, joint_action, is_pad, gripper_action ...]

    Returns
    -------
    train_loader, val_loader
    """

    file_list = []
    for file in glob.glob(os.path.join(dataset_dir, "*.pickle")):
        if task_filter_key is None:
            file_list.append(file)
        else:
            if task_filter_key.search(file):
                file_list.append(file)
    random.shuffle(file_list)

    split_idx = int(len(file_list) * train_split)
    # print("Sampling train dataset from a beta distribution: 1.5, 1.5")
    train_dataset = ReverseTrajDataset(
        file_list=file_list[:split_idx],
        required_data_keys=required_data_keys,
        chunk_size=chunk_size,
        norm_bound=norm_bound,
        add_task_ind=add_task_ind
        # sampler=partial(np.random.beta, 1.5, 1.5),
    )
    val_dataset = ReverseTrajDataset(
        file_list=file_list[split_idx:],
        required_data_keys=required_data_keys,
        chunk_size=chunk_size,
        norm_bound=norm_bound,
        add_task_ind=add_task_ind,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=1,
        prefetch_factor=1,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=1,
        prefetch_factor=1,
    )

    return train_loader, val_loader
