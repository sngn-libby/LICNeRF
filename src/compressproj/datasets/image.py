# Copyright (c) 2021-2022, InterDigital Communications, Inc
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from pathlib import Path

from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset

#from kmeans_pytorch import kmeans
from kmeans_gpu import KMeans as kmeans
from einops import rearrange


class ImageFolderDEcluster(Dataset):
    """Load an image folder database. Training and testing image samples
    are respectively stored in separate directories:

    .. code-block::

        - rootdir/
            - train/
                - img000.png
                - img001.png
	    - train_depth/
            - test/
                - img000.png
                - img001.png
	    - test_depth/

    Args:
        root (string): root directory of the dataset
        transform (callable, optional): a function or transform that takes in a
            PIL image and returns a transformed version
        split (string): split mode ('train' or 'val')
    """

    def __init__(self, root, transform=None, split="train", patch_size=(256, 256)):
        splitdir = Path(root) / split
        depthsplit = split + "_depth"
        depthdir = Path(root) / depthsplit

        if not splitdir.is_dir():
            raise RuntimeError(f'Invalid directory "{root}"')

        self.patch_size = patch_size
        self.samples = [f for f in splitdir.iterdir() if f.is_file()]
        self.depth_samples = [f for f in depthdir.iterdir() if f.is_file()]
        print(f":: DEBUG :: samples #{len(self.samples)}, depth samples #{len(self.depth_samples)}")

        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            img: `PIL.Image.Image` or transformed `PIL.Image.Image`.
        """
        img = Image.open(self.samples[index]).convert("RGB")
        if len(self.depth_samples) <= index:
            #print(f":: DEBUG :: {self.samples[index]}")
            depth = img.getchannel('R')
            depth = depth.crop((0, 0, *self.patch_size))
        else:
            depth = np.load(self.depth_samples[index])
        if self.transform:
            img = self.transform(img)
            if len(self.depth_samples) <= index:
                depth = self.transform(depth)
            else:
                depth = torch.from_numpy(depth)
            
            # clustering
            print(f"image shape: {img.shape}, depth shape: {depth.shape}")
            data = torch.cat([img, depth], dim=1)
            print(f"integrated1 : {data.shape}")
            b, c, w, h = data.shape
            data_px = data.clone().reshape(b, c, -1)
            data_px = rearrange(data_px, 'b c wh -> b wh c')
            first_cluster_idx = kmeans(X=data_px[0], num_clusters=30, distance='cosine').unsqueeze(dim=0)
            for one_img in data_px[1:]: # data_px[1], ... data_px[b-1]
                one_cluster_idx = kmeans(X=one_img, num_clusters=30, distance='cosine').unsqueeze(dim=0)
                cluster_idx = torch.cat([first_cluster_idx, one_cluster_idx], dim=0) # [b, wh, c]
            cluster_idx = rearrange(cluster_idx, 'b wh c -> b c wh')    
            cluster_idx = cluster_idx.reshape(b, 1, w, h)
            clustered_img = torch.cat([data, cluster_idx], dim=1)
            print(f"integrated2 : {clustered_img.shape}")
            
            return clustered_img

        return img

    def __len__(self):
        return len(self.samples)
        

class ImageFolderDE(Dataset):
    """Load an image folder database. Training and testing image samples
    are respectively stored in separate directories:

    .. code-block::

        - rootdir/
            - train/
                - img000.png
                - img001.png
	    - train_depth/
            - test/
                - img000.png
                - img001.png
	    - test_depth/

    Args:
        root (string): root directory of the dataset
        transform (callable, optional): a function or transform that takes in a
            PIL image and returns a transformed version
        split (string): split mode ('train' or 'val')
    """

    def __init__(self, root, transform=None, split="train", patch_size=(256, 256)):
        splitdir = Path(root) / split
        depthsplit = split + "_depth"
        depthdir = Path(root) / depthsplit

        if not splitdir.is_dir():
            raise RuntimeError(f'Invalid directory "{root}"')

        self.patch_size = patch_size
        self.samples = [f for f in splitdir.iterdir() if f.is_file()]
        self.depth_samples = [f for f in depthdir.iterdir() if f.is_file()]
        print(f":: DEBUG :: samples #{len(self.samples)}, depth samples #{len(self.depth_samples)}")

        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            img: `PIL.Image.Image` or transformed `PIL.Image.Image`.
        """
        img = Image.open(self.samples[index]).convert("RGB")
        if len(self.depth_samples) <= index:
            #print(f":: DEBUG :: {self.samples[index]}")
            depth = img.getchannel('R')
            depth = depth.crop((0, 0, *self.patch_size))
        else:
            depth = np.load(self.depth_samples[index])
        if self.transform:
            img = self.transform(img)
            if len(self.depth_samples) <= index:
                depth = self.transform(depth)
            else:
                depth = torch.from_numpy(depth)
            
            return torch.cat([img, depth], dim=0)

        return img

    def __len__(self):
        return len(self.samples)



class ImageFolder(Dataset):
    """Load an image folder database. Training and testing image samples
    are respectively stored in separate directories:

    .. code-block::

        - rootdir/
            - train/
                - img000.png
                - img001.png
            - test/
                - img000.png
                - img001.png

    Args:
        root (string): root directory of the dataset
        transform (callable, optional): a function or transform that takes in a
            PIL image and returns a transformed version
        split (string): split mode ('train' or 'val')
    """

    def __init__(self, root, transform=None, split="train"):
        splitdir = Path(root) / split

        if not splitdir.is_dir():
            raise RuntimeError(f'Invalid directory "{root}"')

        self.samples = [f for f in splitdir.iterdir() if f.is_file()]

        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            img: `PIL.Image.Image` or transformed `PIL.Image.Image`.
        """
        img = Image.open(self.samples[index]).convert("RGB")
        if self.transform:
            return self.transform(img)
        return img

    def __len__(self):
        return len(self.samples)
