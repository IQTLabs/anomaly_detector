#!/usr/bin/env python3

"""
   Copyright 2023 IQT Labs LLC

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

import tqdm
import torch
import torchvision
from PIL import Image
from pathlib import Path
import sklearn.decomposition

class AnomalyDetector:
    """
    ~~~ Computer Vision Anomaly Detection ~~~
    Based on: Napoletano, Piccoli, and Schettini.  Anomaly Detection in
    Nanofibrous Materials by CNN-Based Self-Similarity.  Sensors, 2018, 209.
    """
    def __init__(self,
                 tile=None, tile_height=None, tile_width=None,
                 stride=None, stride_height=None, stride_width=None,
                 device=None, parallel=True, device_ids=None,
                 pca_variance=0.95, threshold=None, verbose=0):

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
        self.device = torch.device(device if device is not None
                                   else 'cuda:0' if torch.cuda.is_available()
                                   else 'cpu')
        self.cnn = self.cnn.to(self.device)
        if parallel and torch.cuda.device_count() > 1 \
           and self.device != torch.device('cpu'):
            self.cnn = torch.nn.DataParallel(self.cnn, device_ids)
        self.cnn.eval()

        # Principal component analysis
        self.pca = sklearn.decomposition.PCA(n_components=pca_variance,
                                             copy=True)

        self.threshold = threshold
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

    def return_features(self, files: list, batchsize: int = 1024) -> torch.Tensor:
        """
        Tile the image in each file, generate a feature vector for each tile,
        and return the feature vectors for all images together in a tensor.
        """
        transform_tile = torchvision.transforms.Compose([
            torchvision.transforms.PILToTensor(),
            lambda x: x.to(self.device),
            lambda x: x.unfold(1, self.tile_height, self.stride_height),
            lambda x: x.unfold(2, self.tile_width, self.stride_width),
            lambda x: x.permute(1, 2, 0, 3, 4),
            lambda x: x.reshape(-1, *x.shape[-3:]),
        ])
        transform_format = torchvision.transforms.Compose([
            lambda x: x / 255.,
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
            torchvision.transforms.Resize(
                size=(self.cnn_input_height, self.cnn_input_width),
                interpolation=torchvision.transforms.InterpolationMode.BICUBIC
            ),
        ])
        features = None
        for filepath in tqdm.tqdm(files, desc='Files'):
            im = Image.open(filepath).convert('RGB')
            img = transform_tile(im)
            for batch in tqdm.tqdm(torch.split(img, batchsize, dim=0),
                                   desc='Tiles'):
                batch_resized = transform_format(batch).float()
                with torch.set_grad_enabled(False):
                    partial_features = self.cnn(batch_resized).detach().cpu()
                features = partial_features if features is None else \
                           torch.cat((features, partial_features), dim=0)
        return features

    def train(self, train_img_dir: Path = None, val_img_dir: Path = None,
              auto_threshold=True):

        train_files = self.return_files(train_img_dir)
        train_features = self.return_features(train_files)

        if auto_threshold:
            val_files = self.return_files(val_img_dir)
            val_features = self.return_features(val_files)

if __name__ == '__main__':
    ad = AnomalyDetector()
    ad.train(Path('../dataset/train'), Path('../dataset/val'))
