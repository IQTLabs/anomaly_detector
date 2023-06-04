#!/usr/bin/env python3

"""
   Copyright 2013 IQT Labs LLC

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

import torch
import torchvision
from PIL import Image
from pathlib import Path

class AnomalyDetector:
    """
    ~~~ Computer Vision Anomaly Detection ~~~
    Based on: Napoletano, Piccoli, and Schettini.  Anomaly Detection in
    Nanofibrous Materials by CNN-Based Self-Similarity.  Sensors, 2018, 209.
    """
    def __init__(self,
                 tile=None, tile_height=None, tile_width=None,
                 stride=None, stride_height=None, stride_width=None,
                 verbose=0):

        # Tile (i.e., patch) size and stride
        tile_default = 32
        stride_default = 8
        non = lambda x: x is not None
        self.tile_height = next(filter(non, [tile_height, tile, tile_default]))
        self.tile_width = next(filter(non, [tile_width, tile, tile_default]))
        self.stride_height = next(filter(non, [
            stride_height, stride, stride_default]))
        self.stride_width = next(filter(non, [
            stride_width, stride, stride_default]))

        # Convolutional neural net
        self.cnn = torchvision.models.resnet18(pretrained=True)
        self.cnn_input_height = 320
        self.cnn_input_width = 320

        self.verbose = verbose

    def return_files(self, img_dir: Path) -> list:
        """
        Takes directory or file path and returns list of file paths
        """
        if img_dir.is_file():
            file_list = [img_dir]
        elif img_dir.is_dir():
            file_list = [x for x in img_dir.glob('**/*') if x.is_file()]
        if self.verbose >= 2:
            print('return_files:', file_list)
        return file_list

    def return_features(self, files: list):
        """
        Tile the image in each file, generate a feature vector for each tile,
        and return the feature vectors for all images together in a tensor.
        """
        transform = torchvision.transforms.Compose([
            torchvision.transforms.PILToTensor(),
            lambda x: x.unfold(1, self.tile_height, self.stride_height),
            lambda x: x.unfold(2, self.tile_width, self.stride_width),
            lambda x: x.permute(1, 2, 0, 3, 4),
            lambda x: x.reshape(-1, *x.shape[-3:]),
            torchvision.transforms.Resize(
                size=(self.cnn_input_height, self.cnn_input_width),
                interpolation=torchvision.transforms.InterpolationMode.BICUBIC
            ),
        ])
        for filepath in files:
            im = Image.open(filepath).convert('RGB')
            img = transform(im)
            print(im.size)
            print(type(img))
            print(img.size())
        return None

    def train(self, train_img_dir: Path = None, val_img_dir: Path = None):

        train_files = self.return_files(train_img_dir)
        val_files = self.return_files(val_img_dir)
        train_features = self.return_features(train_files)

if __name__ == '__main__':
    ad = AnomalyDetector()
    ad.train(Path('../dataset/train'), Path('../dataset/val'))
