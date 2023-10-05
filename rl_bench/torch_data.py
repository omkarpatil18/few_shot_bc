import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import glob
import os
import random

CONFIG_DIM = 7  # joint space


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


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
    Parameters
    ---------
    file_list: default=[]
        List of file episodes
    required_data_keys: default=[]
        Keys to extract from the recorded demonstrations
    chunk_size: default=10,
        Chunk of action size to return
    norm_bound: default==None,
        Bounds for normalizing the data

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

    Keys returned are:
        [images, joint_action, is_pad, gripper_action ...]
    """

    task_filter_map = {"box": 1, "door": 2}
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
    ):
        self.file_list = file_list
        self.required_data_keys = required_data_keys
        self.chunk_size = chunk_size
        if norm_bound is not None:
            self.normalizer = ConfigMinMaxNormalization(norm_bound)
        else:
            self.normalizer = None
        self.len = len(self.file_list)

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
            start_ts = np.random.choice(episode_len)
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
                    # Normalize images?
                    image_dict[key] = demo[key][start_ts]
                else:
                    data = demo[key][start_ts]
                    data = torch.as_tensor(data)
                    data_batch[key] = data

            # traj_type = self.file_list[index].split("_")[-2]
            # if traj_type == "forward":
            #     latent = np.array([start_ts / episode_len, 1.0, 0.0], dtype=np.float32)
            # elif traj_type == "backward":
            #     latent = np.array([start_ts / episode_len, 0.0, 1.0], dtype=np.float32)
            # data_batch.append(torch.as_tensor(latent))

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

        assert data_batch["images"].shape == torch.Size([4, 3, 128, 128])
        assert data_batch["is_pad"].shape == torch.Size([100])
        assert data_batch["joint_action"].shape == torch.Size([100, 7])
        assert data_batch["gripper_action"].shape == torch.Size([100])

        return data_batch


def load_data(
    dataset_dir,
    task_filter_key=ReverseTrajDataset.task_filter_map["box"],
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
):
    file_list = []
    for file in glob.glob(os.path.join(dataset_dir, "*.pickle")):
        if task_filter_key is None:
            file_list.append(file)
        else:
            task_type_key = int(file.split(".")[-2].split("_")[-1]) % 10
            if task_type_key == task_filter_key:
                file_list.append(file)
    random.shuffle(file_list)

    split_idx = int(len(file_list) * train_split)
    train_dataset = ReverseTrajDataset(
        file_list=file_list[:split_idx],
        required_data_keys=required_data_keys,
        chunk_size=chunk_size,
        norm_bound=norm_bound,
    )
    val_dataset = ReverseTrajDataset(
        file_list=file_list[split_idx:],
        required_data_keys=required_data_keys,
        chunk_size=chunk_size,
        norm_bound=norm_bound,
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
