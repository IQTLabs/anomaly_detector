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


class AnomalyDetector:
    """
    ~~~ Computer Vision Anomaly Detection ~~~
    Based on: Napoletano, Piccoli, and Schettini.  Anomaly Detection in
    Nanofibrous Materials by CNN-Based Self-Similarity.  Sensors, 2018, 209.
    """
    def __init__(self,
                 tile=None, tile_height=None, tile_width=None,
                 stride=None, stride_height=None, stride_width=None):

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


    def train():
        pass

if __name__ == '__main__':
    ad = AnomalyDetector(tile=77, tile_height=88)
