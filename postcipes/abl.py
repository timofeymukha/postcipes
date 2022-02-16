# This file is part of postcipes
# (c) Timofey Mukha
# The code is released under the MIT Licence.
# See LICENCE.txt and the Legal section in the README for more information

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pyparsing import unicode_set

from .postcipe import Postcipe
import turbulucid as tbl
import numpy as np
import h5py
from os.path import join
from scipy.integrate import simps
from scipy.interpolate import interp1d

__all__ = ["ABL"]


class ABL(Postcipe):

    def __init__(self, path=None, nu=None, uStar=1, z0=None, time=None, readH=False, initEmpty=False):
        Postcipe.__init__(self)
        if initEmpty:
            return
        self.case = path
        self.readPath = join(self.case, "postProcessing", "collapsedFields",
                             str(time))
        self.nu = nu
        self.time = time
        self.uStar = uStar
        self.z0 = z0

        self.y = np.genfromtxt(join(self.readPath, "UMean_X.xy"))[:, 0]
        self.u = np.genfromtxt(join(self.readPath, "UMean_X.xy"))[:, 1]
        self.v = np.genfromtxt(join(self.readPath, "UMean_Y.xy"))[:, 1]
        self.w = np.genfromtxt(join(self.readPath, "UMean_Z.xy"))[:, 1]
        self.uu = np.genfromtxt(join(self.readPath, "UPrime2Mean_XX.xy"))[:, 1]
        self.vv = np.genfromtxt(join(self.readPath, "UPrime2Mean_YY.xy"))[:, 1]
        self.ww = np.genfromtxt(join(self.readPath, "UPrime2Mean_ZZ.xy"))[:, 1]
        self.uv = np.genfromtxt(join(self.readPath, "UPrime2Mean_XY.xy"))[:, 1]
        self.k = 0.5*(self.uu + self.vv + self.ww)
        self.nut = np.genfromtxt(join(self.readPath, "nutMean.xy"))[:, 1]

        tau = np.genfromtxt(join(self.readPath,
                                 "wallShearStressMean_X.xy"))[:, 1]
        self.tau = tau[0]

        self.delta = self.y[-1]
        self.uB = simps(self.u, self.y)/self.delta
        self.uC = self.u[-1]

        self.yStar = self.y/self.z0
        self.uPlus = self.u/self.uStar
        self.uuPlus = self.uu/self.uStar**2
        self.vvPlus = self.vv/self.uStar**2
        self.wwPlus = self.ww/self.uStar**2
        self.uvPlus = self.uv/self.uStar**2
        self.kPlus = self.k/self.uStar**2
        self.uRms = np.sqrt(self.uu)/self.uStar
        self.vRms = np.sqrt(self.vv)/self.uStar
        self.wRms = np.sqrt(self.ww)/self.uStar

        self.yf = self.cell_faces()
        if readH:
            self.h = np.genfromtxt(join(self.readPath,
                                        "h.xy"))[:, 1]

    def cell_faces(self):
        faces = np.zeros(self.y.size - 1)
        faces[0] = self.y[0]
        faces[-1] = self.y[-1]

        for i in range(faces.size):
            if i == 0 or i == faces.size -1:
                continue
            else:
                faces[i] = faces[i - 1] + (self.y[i] - faces[i - 1])*2
                # Ensure delta faces is set exact
                if np.abs(faces[i] - self.delta)/self.delta < 0.01:
                    faces[i] = self.delta

        return faces

    def save(self, name):
        f = h5py.File(name, 'w')

        f.attrs["uStar"] = self.uStar
        f.attrs["z0"] = self.z0
        f.attrs["uB"] = self.uB
        f.attrs["uC"] = self.uC
        f.attrs["delta"] = self.delta

        f.create_dataset("y", data=self.y)
        f.create_dataset("u", data=self.u)
        f.create_dataset("uu", data=self.uu)
        f.create_dataset("vv", data=self.vv)
        f.create_dataset("ww", data=self.ww)
        f.create_dataset("k", data=self.k)
        f.create_dataset("uv", data=self.uv)
        f.create_dataset("nut", data=self.nut)
        f.create_dataset("yStar",data=self.yStar)
        f.create_dataset("uStar", data=self.uStar)
        f.create_dataset("uuPlus", data=self.uuPlus)
        f.create_dataset("vvPlus", data=self.vvPlus)
        f.create_dataset("wwStar", data=self.wwStar)
        f.create_dataset("uvStar", data=self.uvStar)
        f.create_dataset("kStar", data=self.kStar)
        f.create_dataset("uRms", data=self.uRms)
        f.create_dataset("vRms", data=self.vRms)
        f.create_dataset("wRms", data=self.wRms)

        f.close()

    def load(self, name):
        f = h5py.File(name, 'r')

        self.nu = f.attrs["nu"]
        self.uStar = f.attrs["uStar"]
        self.uB = f.attrs["uB"]
        self.uC = f.attrs["uC"]
        self.delta = f.attrs["delta"]
        self.reB = f.attrs["reB"]
        self.reC = f.attrs["reC"]

        self.y = f["y"][:]
        self.u = f["u"][:]
        self.uu = f["uu"][:]
        self.vv = f["vv"][:]
        self.ww = f["ww"][:]
        self.k = f["k"][:]
        self.uv = f["uv"][:]
        self.nut = f["nut"][:]
        self.yStar = f["yStar"][:]
        self.uPlus = f["uStar"][:]
        self.uuPlus= f["uuPlus"][:]
        self.vvPlus = f["vvPlus"][:]
        self.wwPlus = f["wwStar"][:]
        self.uvPlus = f["uvStar"][:]
        self.uvPlus = f["kStar"][:]
        self.uRms = f["uRms"][:]
        self.vRms = f["vRms"][:]
        self.vRms = f["wRms"][:]
        self.kPlus = f["kStar"][:]

        f.close()
