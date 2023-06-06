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

import math
import tqdm
import torch
import torchvision
import numpy as np
from PIL import Image
from pathlib import Path
import sklearn.decomposition
import sklearn.preprocessing
import sklearn.cluster

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
                 cnn_batchsize=1024, pca_variance=0.95,
                 kmeans_clusters=10, kmeans_trials=10, kmeans_neighbors=1,
                 threshold=None, zscore=1.645, cdf=0.95,
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
        self.device = torch.device(device if device is not None
                                   else 'cuda:0' if torch.cuda.is_available()
                                   else 'cpu')
        self.cnn = self.cnn.to(self.device)
        if parallel and torch.cuda.device_count() > 1 \
           and self.device != torch.device('cpu'):
            self.cnn = torch.nn.DataParallel(self.cnn, device_ids)
        self.cnn.eval()
        self.cnn_batchsize = cnn_batchsize

        # Principal component analysis
        self.pca = sklearn.decomposition.PCA(
            n_components=pca_variance, copy=True)
        self.pca_scaler = sklearn.preprocessing.StandardScaler()

        # K-Means clustering
        self.kmeans = sklearn.cluster.KMeans(
            n_clusters=kmeans_clusters, init='k-means++', n_init=kmeans_trials,
            verbose=max(0, verbose - 1))
        self.kmeans_neighbors = kmeans_neighbors

        self.threshold = threshold
        self.zscore = zscore
        self.cdf = cdf
        self.verbose = verbose

    def return_files(self, img_dir: Path) -> list:
        """
        Takes directory or file path and returns list of file paths
        """
        if img_dir.is_file():
            file_list = [img_dir]
        elif img_dir.is_dir():
            file_list = [x for x in img_dir.glob('**/*') if x.is_file()]
            file_list = sorted(file_list)
        if self.verbose >= 2:
            print('return_files:', file_list)
        return file_list

    def return_features(self, files: list, metadata: bool = False) -> torch.Tensor:
        """
        Tile the image in each file, generate a feature vector for each tile,
        and return the feature vectors for all images together in a tensor.
        """
        transform_tile = torchvision.transforms.Compose([
            torchvision.transforms.PILToTensor(),
            lambda x: x.to(self.device).float(),
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
        metas = []
        for filepath in tqdm.tqdm(files, desc='Files'):
            im = Image.open(filepath).convert('RGB')
            img = transform_tile(im)
            if metadata:
                metas.append({'path': filepath,
                              'height': im.height,
                              'width': im.width,
                              'tiles': img.size(0)})
            for batch in tqdm.tqdm(torch.split(img, self.cnn_batchsize, dim=0),
                                   desc='Tiles'):
                batch_resized = transform_format(batch)
                with torch.set_grad_enabled(False):
                    partial_features = self.cnn(batch_resized).detach().cpu()
                features = partial_features if features is None else \
                           torch.cat((features, partial_features), dim=0)
        if metadata:
            return features, metas
        else:
            return features

    def reduce_features(self, features: torch.Tensor, fit: bool = False) -> torch.Tensor:
        """
        Optionally fit principal component analysis (PCA) model,
        and use it to reduce dimensionality of feature vectors.
        Features are normalized after dimensionality reduction.
        """
        features = features.numpy()
        if fit:
            features = self.pca.fit_transform(features)
            features = self.pca_scaler.fit_transform(features)
        else:
            features = self.pca.transform(features)
            features = self.pca_scaler.transform(features)
        #features = torch.from_numpy(features)
        return features

    def return_distances(self, features: np.ndarray, fit: bool = False) -> np.ndarray:
        """
        Optionally fit k-means clustering centroids, and return distance
        to nearest centroid from each feature vector.
        """
        if fit:
            distances = self.kmeans.fit_transform(features)
        else:
            distances = self.kmeans.transform(features)
        if self.kmeans_neighbors == 1:
            distances = np.amin(distances, axis=1)
        else:
            distances = np.sort(distances, axis=1)
            distances = np.mean(distances[:, :self.kmeans_neighbors], axis=1)
        return distances

    def set_threshold_zscore(self, distances: np.ndarray) -> float:
        """
        Set threshold for anomalies using z-score
        """
        mean = np.mean(distances)
        stdev = np.std(distances)
        threshold = mean + self.zscore * stdev
        if self.verbose >= 1:
            print('Threshold stats:')
            print('    Mean  :', mean)
            print('    StDev :', stdev)
            print('    Thresh:', threshold)
        self.threshold = threshold
        return threshold

    def set_threshold_cdf(self, distances: np.ndarray) -> float:
        """
        Set threshold for anomalies using cumulative distribution function
        """
        count = np.size(distances)
        distances = np.sort(distances)
        threshold_index = min(math.floor(self.cdf * count), count - 1)
        threshold = distances[threshold_index]
        if self.verbose >= 1:
            print('Threshold:', threshold)
        self.threshold = threshold
        return threshold

    def return_pixelmaps(self, data: np.ndarray, metas: list, img_dir: Path):
        """
        Map per-tile data values back to image pixels for multiple images,
        using averages of overlapping tiles.
        """
        start_idx = 0
        for meta in metas:
            end_idx = start_idx + meta['tiles']
            pixelmap = self.return_pixelmap(data[start_idx:end_idx], meta)
            mask = self.return_mask(pixelmap)
            start_idx = end_idx

            img = Image.fromarray(mask, mode='1')
            path = img_dir / meta['path'].name
            img.save(path)

    def return_pixelmap(self, data: np.ndarray, meta: dict) -> torch.Tensor:
        """
        Map per-tile data values back to image pixels for a single image,
        using averages of overlapping tiles.
        """
        # Unfolded: (batch=1, channels=1, tile_height, tile_width,
        #            count_height, count_width)
        count_height = (meta['height'] - self.tile_height) \
                       // self.stride_height + 1
        count_width = (meta['width'] - self.tile_width) \
                      // self.stride_width + 1
        canvas_unfolded = torch.zeros((
            1, 1, self.tile_height, self.tile_width,
            count_height, count_width))
        datagrid = torch.from_numpy(data).reshape((
            1, 1, 1, 1, count_height, count_width))
        numerator_unfolded = canvas_unfolded + datagrid
        denominator_unfolded = canvas_unfolded + 1
        numerator_unfolded = numerator_unfolded.reshape(1, 1 * self.tile_height * self.tile_width, count_height * count_width)
        denominator_unfolded = denominator_unfolded.reshape(1, 1 * self.tile_height * self.tile_width, count_height * count_width)
        numerator = torch.nn.functional.fold(numerator_unfolded, (meta['height'], meta['width']), (self.tile_height, self.tile_width), stride=(self.stride_height, self.stride_width)).squeeze(0)
        denominator = torch.nn.functional.fold(denominator_unfolded, (meta['height'], meta['width']), (self.tile_height, self.tile_width), stride=(self.stride_height, self.stride_width)).squeeze(0)
        average = torch.nan_to_num(numerator / denominator, nan=0)
        average = average.squeeze().numpy()
        return average

    def return_mask(self, pixelmap):
        """
        Returns binary mask, with zero indiciating normal
        and one indicating an anomaly.
        """
        mask = (pixelmap >= self.threshold)
        return mask

    def anomalous_tiles(self, distances: np.ndarray) -> np.ndarray:
        """
        Identify anomalous tiles.  Note that anomaly percentage
        for tiles will generally not be the same as for pixels.
        """
        flags = (distances >= self.threshold)
        if self.verbose >= 1:
            print('Tile stats: %i out of %i tiles are anomalies (%.2f pct)'
                  % (np.sum(flags), np.size(flags),
                     100 * np.sum(flags) / np.size(flags)))
        return flags

    def train(self, train_img_dir: Path, val_img_dir: Path = None,
              auto_threshold=True):
        """
        Train the model, using folders of training and validation data
        """
        train_files = self.return_files(train_img_dir)
        train_features = self.return_features(train_files)
        train_features = self.reduce_features(train_features, fit=True)
        train_distances = self.return_distances(train_features, fit=True)

        if val_img_dir is not None and auto_threshold:
            val_files = self.return_files(val_img_dir)
            val_features = self.return_features(val_files)
            val_features = self.reduce_features(val_features)
            val_distances = self.return_distances(val_features)
            self.set_threshold_cdf(val_distances)

    def test(self, test_img_dir: Path, output_img_dir: Path = None):
        """
        Run inference with the model
        """
        test_files = self.return_files(test_img_dir)
        test_features, test_metadata = self.return_features(
            test_files, metadata=True)
        test_features = self.reduce_features(test_features)
        test_distances = self.return_distances(test_features)
        self.anomalous_tiles(test_distances)
        if output_img_dir is not None:
            test_pixelmaps = self.return_pixelmaps(
                test_distances, test_metadata, output_img_dir)


if __name__ == '__main__':
    ad = AnomalyDetector(verbose=1)
    ad.train(Path('../dataset/train'), Path('../dataset/val'))
    ad.test(Path('../dataset/test'), Path('../dataset/output'))
