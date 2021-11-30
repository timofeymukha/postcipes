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
from os.path import join

__all__ = ["PeriodicHill"]


class PeriodicHill(Postcipe):
    def __init__(self, path, nu, scale, wallnames, wallmodel):
        Postcipe.__init__(self)

        self.case = tbl.Case(path)
        self.nu = nu
        self.scale = scale
        self.case.scale(scale, scale)
        self.wallmodel=wallmodel

        self.taubot = self.get_tau(wallnames[0])
        self.tautop = self.get_tau(wallnames[1])
        self.cfbot = 2*self.taubot
        self.cftop = 2*self.tautop
        self.utaubot = np.sqrt(np.abs(self.taubot))
        self.utautop = np.sqrt(np.abs(self.tautop))
        self.yplusbot = self.get_yplus(wallnames[0])
        self.yplustop = self.get_yplus(wallnames[1])

        self.xb = self.case.boundary_data(wallnames[0], "x")[0][:, 0]
        self.yb = self.case.boundary_data(wallnames[0], "x")[0][:, 1]
        self.xt = self.case.boundary_data(wallnames[1], "x")[0][:, 0]
        self.yt = self.case.boundary_data(wallnames[1], "x")[0][:, 1]

    def get_tau(self, wall):
        if self.wallmodel:
            return self.get_tau_wmles(wall)
        else:
            return self.get_tau_wrles(wall)

    def get_tau_wmles(self, wall):
        ind = tbl.sort_indices(self.case, wall, "x")

        tau = self.case.boundary_data(wall)[1]["wallShearStressMean"][ind, :2]
        t = tbl.tangents(self.case, wall)[ind]

        tanu = np.zeros(t.shape[0])

        for i in range(tanu.size):
            tanu[i] = np.dot(tau[i, :], t[i, :])
        return tanu

    def get_tau_wrles(self, wall):
        ind = tbl.sort_indices(self.case, wall, "x")
        distance = tbl.dist(self.case, wall, True)[ind]*self.scale
        u = self.case.boundary_cell_data(wall)[1]['UMean'][ind, :2]
        t = tbl.tangents(self.case, wall)[ind]

        tanu = np.zeros(t.shape[0])

        for i in range(tanu.size):
            tanu[i] = np.dot(u[i, :], t[i, :])

        return self.nu * tanu / distance

    def get_yplus(self, wall):
        dist = tbl.dist(self.case, wall, corrected=True, sort="x") * self.scale
        return dist*np.sqrt(np.abs(self.get_tau(wall)))/self.nu

    def compute_ub(self, boundary):
        dy = tbl.edge_lengths(self.case, boundary)
        yin, uin = self.case.boundary_data(boundary)
        uin = uin["UMean"][:, 0]
        return np.sum(uin*dy)/np.sum(dy)

