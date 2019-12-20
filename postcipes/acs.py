# This file is part of postcipes
# (c) Timofey Mukha
# The code is released under the MIT Licence.
# See LICENCE.txt and the Legal section in the README for more information

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from .postcipe import Postcipe
import turbulucid as tbl
import numpy as np
import h5py

__all__ = ["ACS"]


class ACS(Postcipe):

    def __init__(self, path):
        Postcipe.__init__(self)
        self.case = tbl.Case(path)
        self.dp = np.abs(np.mean(self.case.boundary_data("airInlet")[1]['p_rgh']))
        self.u0 = np.mean(self.case.boundary_data("waterInlet")[1]['U'][:, 0])
        self.sigma = 0.07

        self.nuAir = 1.48e-05
        self.nuWater = 1.139e-06
        self.rhoWater = 999
        self.rhoAir = 2.5
        self.muAir = self.nuAir*self.rhoAir
        self.muWater = self.nuWater*self.rhoWater

        g = 9.81

        self.lambdaLinear = 2*np.pi*self.u0**2/g
        self.amplitudeLinear = self.dp/(self.rhoWater*g)*np.sqrt(2)

        self.alpha05 = tbl.isoline(self.case, 'alpha.waterMean', 0.5)

        ind = np.where(self.alpha05[:, 0] < 4)
        self.crest = np.max(self.alpha05[ind, 1])
        self.trough = np.min(self.alpha05[ind, 1])
        self.y0 = 0.5*(self.crest + self.trough)
        self.amplitude = self.crest - self.trough

        indCrest = np.argmax(self.alpha05[ind, 1])
        indTrough = np.argmin(self.alpha05[ind, 1])
        self.xCrest = self.alpha05[ind, 0][0, indCrest]
        self.xTrough = self.alpha05[ind, 0][0, indTrough]

        self.lambdaReal = 2*np.abs(self.xTrough - self.xCrest)

        self.Ca = self.dp/(0.5*self.rhoWater*self.u0**2)
        self.Fn = self.u0/np.sqrt(g*self.lambdaReal)
        self.We = self.rhoWater*self.u0**2*self.lambdaReal/self.sigma

        self.case['k'] = 0.5*np.sum(self.case['UPrime2Mean'][:, :2], axis=1)
