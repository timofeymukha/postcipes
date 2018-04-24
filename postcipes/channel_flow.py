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
from os.path import join
from scipy.integrate import simps

__all__ = ["ChannelFlow"]


class ChannelFlow(Postcipe):

    def __init__(self, path, nu, time, wallModel=False):
        Postcipe.__init__(self)
        self.case = path
        self.readPath = join(self.case, "postProcessing", "collapsedFields",
                             str(time))
        self.nu = nu
        self.time = time

        self.y = np.genfromtxt(join(self.readPath, "UMean_X.xy"))[:, 0]
        self.u = np.genfromtxt(join(self.readPath, "UMean_X.xy"))[:, 1]
        self.uu = np.genfromtxt(join(self.readPath, "UPrime2Mean_XX.xy"))[:, 1]
        self.vv = np.genfromtxt(join(self.readPath, "UPrime2Mean_YY.xy"))[:, 1]
        self.ww = np.genfromtxt(join(self.readPath, "UPrime2Mean_ZZ.xy"))[:, 1]
        self.uv = np.genfromtxt(join(self.readPath, "UPrime2Mean_XY.xy"))[:, 1]
        self.k = 0.5*(self.uu + self.vv + self.ww)
        self.nut = np.genfromtxt(join(self.readPath, "nutMean.xy"))[:, 1]

        self.tau = 0
        if wallModel:
            tau = np.genfromtxt(join(self.readPath,
                                     "wallShearStressMean_X.xy"))[:, 1]
            self.tau = 0.5*(tau[0] + tau[-1])
        else:
            self.tau = nu*0.5*(self.u[1] + self.u[-2])/self.y[1]

        self.uTau = np.sqrt(self.tau)
        self.delta = 0.5*(self.y[-1] - self.y[0])
        self.uB = simps(self.u, self.y)/(2*self.delta)
        self.uC = 0.5*(self.u[int(self.y.size/2)] +
                       self.u[int(self.y.size/2) -1])

        self.yPlus = self.y*self.uTau/self.nu
        self.uPlus = self.u/self.uTau
        self.uuPlus = self.uu/self.uTau**2
        self.vvPlus = self.vv/self.uTau**2
        self.wwPlus = self.ww/self.uTau**2
        self.uvPlus = self.uv/self.uTau**2
        self.kPlus = self.k/self.uTau**2
        self.uRms = np.sqrt(self.uu)/self.uTau
        self.vRms = np.sqrt(self.vv)/self.uTau
        self.wRms = np.sqrt(self.ww)/self.uTau

        self.reTau = self.uTau*self.delta/self.nu
        self.reB = self.uB*self.delta/self.nu
        self.reC = self.uC*self.delta/self.nu

        self.theta = tbl.momentum_thickness(self.y[:int(self.y.size/2)],
                                            self.u[:int(self.u.size/2)],
                                            interpolate=True)
        self.delta99 = tbl.delta_99(self.y[:int(self.y.size/2)],
                                    self.u[:int(self.u.size/2)],
                                    interpolate=True)

        self.deltaStar = tbl.delta_star(self.y[:int(self.y.size/2)],
                                        self.u[:int(self.u.size/2)],
                                        interpolate=True)

        self.reTheta = self.theta*self.uC/self.nu
        self.reDelta99 = self.delta99*self.uC/self.nu
        self.reDeltaStar = self.deltaStar*self.uC/self.nu
