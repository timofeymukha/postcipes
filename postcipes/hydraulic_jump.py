# This file is part of postcipes
# (c) Timofey Mukha
# The code is released under the MIT Licence.
# See LICENCE.txt and the Legal section in the README for more information

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from .postcipe import Postcipe
import turbulucid as tbl
from scipy.interpolate import interp1d
import numpy as np
import h5py

__all__ = ["HydraulicJump"]


class HydraulicJump(Postcipe):

    def __init__(self, path):
        Postcipe.__init__(self)
        self.case = tbl.Case(path)
        self.case['alphag'] = 1 - self.case['alpha.waterMean']
        self.U = self.case.boundary_data("inlet", sort="y")[1]['U'][0, 0]

        alpha_inlet = self.case.boundary_data("inlet", sort="y")[1]['alpha.water']
        y_inlet = self.case.boundary_data("inlet", sort="y")[0][:, 1]
        #alpha_interp = interp1d(y_inlet, alpha_inlet)

        #y_inlet_new = np.linspace(y_inlet[0], y_inlet[-1], 10000)
        #alpha_inlet_new = alpha_interp(y_inlet_new)
        inlet_edge_length = tbl.edge_lengths(self.case, "inlet")
        self.d = y_inlet[-1] + 0.5*inlet_edge_length[-1]
        self.Fr1 = self.U/np.sqrt(9.81*self.d)
        self.d2 = self.d*(np.sqrt(1 + 8*self.Fr1**2) - 1)/2
        self.Fr2 = self.U/np.sqrt(9.81*self.d2)

        iso05 = tbl.isoline(self.case, "alpha.waterMean", 0.5)
        idx = iso05[:, 0].argsort()
        self.xfs = iso05[idx, 0]
        self.yfs = iso05[idx, 1]

        idx_toe = np.argmin(np.abs(self.d*1.1 - self.yfs[:int(self.yfs.size/2)]))
        self.xtoe = self.xfs[idx_toe]
