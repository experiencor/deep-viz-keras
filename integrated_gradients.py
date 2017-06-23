# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities to compute an IntegratedGradients SaliencyMask."""

import numpy as np
from saliency import GradientSaliency

class IntegratedGradients(GradientSaliency):
    """A SaliencyMask class that implements the integrated gradients method.

    https://arxiv.org/abs/1703.01365
    """

    def GetMask(self, input_image, input_baseline=None, nsamples=100):
        """Returns a integrated gradients mask."""
        if input_baseline == None:
            input_baseline = np.zeros_like(input_image)

        assert input_baseline.shape == input_image.shape

        input_diff = input_image - input_baseline

        total_gradients = np.zeros_like(input_image)

        for alpha in np.linspace(0, 1, nsamples):
            input_step = input_baseline + alpha * input_diff
            total_gradients += super(IntegratedGradients, self).get_mast(input_step)

        return total_gradients * input_diff
