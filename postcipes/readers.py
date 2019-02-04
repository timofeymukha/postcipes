# This file is part of postcipes
# (c) Timofey Mukha
# The code is released under the MIT Licence.
# See LICENCE.txt and the Legal section in the README for more information

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from os.path import join
import os
import numpy as np
import h5py

__all__ = ["SwakPatchExpression"]


class SwakPatchExpression:

    def __init__(self, path):

        times = np.array(os.listdir(path))
        timesFloat = [float(i) for i in times]
        times = times[np.argsort(timesFloat)]
        self.times = times
        self.patches = os.listdir(join(path, times[0]))
        data = {}
        for patch in self.patches:
            patchData = np.genfromtxt(join(path, times[0], patch))
            for time in times[1:]:
                patchData = np.append(patchData, np.genfromtxt(join(path, time, patch)), axis=0)

            ind = np.unique(patchData[:, 0], return_index=True)[1]
            data[patch] = patchData[ind, :]

        self.data = data
        self.time = data[self.patches[0]][:, 0]

    def __getitem__(self, item):
        return self.data[item]

