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

from anomaly import AnomalyDetector

ad = AnomalyDetector()
ad.train('example_dataset/train', 'example_dataset/val')
ad.save('example_dataset/model.pickle')

ad = AnomalyDetector.load('example_dataset/model.pickle')
ad.test('example_dataset/test', 'example_dataset/output')
