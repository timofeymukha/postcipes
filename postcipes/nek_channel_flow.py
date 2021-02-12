# This file is part of postcipes
# (c) 2021 Timofey Mukha
# The code is released under the MIT Licence.
# See LICENCE.txt and the Legal section in the README for more information

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from .postcipe import Postcipe
import numpy as np
import h5py
import pymech
import pymech.dataset as ds
from os.path import join
from scipy.interpolate import interp1d
import glob
from scipy.special import roots_jacobi

__all__ = ["NekChannelFlow", "gll", "gll_int"]


def gll_int(f, a, b):
    """Integrate f from a to b using its values at gll points."""
    n = f.size
    x, w = gll(n)
    return 0.5*(b-a)*np.sum(f*w)


def gll(n):
    """Compute the points and weights of the Gouss-Lenendre-Lobbato quadrature

    Parameters
    ----------
        n, int
        The number of points.

    """
    # Special case
    if n == 2:
        return np.array([-1, 1]), np.array([1, 1])

    x, w = roots_jacobi(n - 2, 1, 1)
    for i in range(x.size):
        w[i] /= 1 - x[i] ** 2

    x = np.append(-1, np.append(x, 1))
    w = np.append(2 / (n * (n - 1)), np.append(w, 2 / (n * (n - 1))))
    return x, w


class NekChannelFlow(Postcipe):

    def __init__(self, path=None, basename=None, starttime=0, lx1=8, nu=None):
        Postcipe.__init__(self)

        self.case = path
        self.basename = basename
        self.nu = nu
        datafiles = glob.glob(self.case + '\\sts' + basename +'[0-1].f[0-9][0-9][0-9][0-9][0-9]')
        datasets = [ds.open_dataset(i) for i in datafiles]

        if len(datasets) is 0:
            raise FileExistsError

        namemap = {"s01": "u", "s05": "uu", "s06": "vv", "s07": "ww", "s09": "uv"}

        ndatasets = len(datasets)
        nx = datasets[0].x.size
        nelx = nx / lx1

        integral = {}
        for key in ["s01", "s05", "s06", "s07", "s09"]:
            integral[key] = np.zeros(datasets[0].y.size)

        for d in datasets:
            if d.time.data < starttime:
                ndatasets -= 1
                continue
            for key in ["s01", "s05", "s06", "s07", "s09"]:
                offset = 0
                for e in range(int(nelx)):
                    ind = np.arange(offset, offset + lx1)
                    a = d.x[offset]
                    b = d.x[offset + lx1 - 1]
                    for j in range(d.y.size):
                        slice = d[key].isel(x=ind, y=j).data[0]
                        integral[key][j] += gll_int(slice, a, b) / (b - a)

                    offset += lx1

        for key in integral:
            integral[key] /= nelx*ndatasets

        self.u = integral["s01"]
        self.uu = integral["s05"] - self.u*self.u
        self.vv = integral["s06"]
        self.ww = integral["s07"]
        self.uv = integral["s09"]

        self.y = datasets[0].y.data

#        self.uTau = np.sqrt(self.tau)
#        self.delta = 0.5*(self.y[-1] - self.y[0])
#        self.uB = simps(self.u, self.y)/(2*self.delta)
#        self.uC = 0.5*(self.u[int(self.y.size/2)] +
#                       self.u[int(self.y.size/2) -1])

#        self.yPlus = self.y*self.uTau/self.nu
#        self.uPlus = self.u/self.uTau
#        self.uuPlus = self.uu/self.uTau**2
#        self.vvPlus = self.vv/self.uTau**2
#        self.wwPlus = self.ww/self.uTau**2
#        self.uvPlus = self.uv/self.uTau**2
#        self.kPlus = self.k/self.uTau**2
#        self.uRms = np.sqrt(self.uu)/self.uTau
#        self.vRms = np.sqrt(self.vv)/self.uTau
#        self.wRms = np.sqrt(self.ww)/self.uTau

#        self.reTau = self.uTau*self.delta/self.nu
#        self.reB = self.uB*self.delta/self.nu
#        self.reC = self.uC*self.delta/self.nu

#        self.theta = tbl.momentum_thickness(self.y[:int(self.y.size/2)],
#                                            self.u[:int(self.u.size/2)],
#                                            interpolate=True)
#        self.delta99 = tbl.delta_99(self.y[:int(self.y.size/2)],
#                                    self.u[:int(self.u.size/2)],
#                                    interpolate=True)

#        self.deltaStar = tbl.delta_star(self.y[:int(self.y.size/2)],
#                                        self.u[:int(self.u.size/2)],
#                                        interpolate=True)

#        self.reTheta = self.theta*self.uC/self.nu
#        self.reDelta99 = self.delta99*self.uC/self.nu
#        self.reDeltaStar = self.deltaStar*self.uC/self.nu
#        if readH:
#            self.h = np.genfromtxt(join(self.readPath,
#                                     "h.xy"))[:, 1]

    def utau_relative_error(self, bench, procent=True, abs=False):
        error = (self.uTau - bench)/bench
        if procent:
            error *= 100
        if abs:
            error = np.abs(error)

        return error



    def u_relative_error(self, benchY, benchU, bound=0.2, procent=True):

        bound = bound*self.delta
        benchInterp = interp1d(np.append(benchY, [1]), np.append(benchU, benchU[-1]))
        wmlesInterp = interp1d(self.y, self.u)

        y = np.linspace(bound, 1, 200)

        error = np.max(np.abs(wmlesInterp(y) - benchInterp(y)))/np.max(benchU)

        if procent:
            error *= 100

        return error

    def save(self, name):
        f = h5py.File(name, 'w')

        f.attrs["nu"] = self.nu
        f.attrs["uTau"] = self.uTau
        f.attrs["uB"] = self.uB
        f.attrs["uC"] = self.uC
        f.attrs["delta"] = self.delta
        f.attrs["delta99"] = self.delta99
        f.attrs["deltaStar"] = self.deltaStar
        f.attrs["theta"] = self.reTheta
        f.attrs["reDelta99"] = self.reDelta99
        f.attrs["reDeltaStar"] = self.reDeltaStar
        f.attrs["reTheta"] = self.reTheta
        f.attrs["reTau"] = self.reTau
        f.attrs["reB"] = self.reB
        f.attrs["reC"] = self.reC

        f.create_dataset("y", data=self.y)
        f.create_dataset("u", data=self.u)
        f.create_dataset("uu", data=self.uu)
        f.create_dataset("vv", data=self.vv)
        f.create_dataset("ww", data=self.ww)
        f.create_dataset("k", data=self.k)
        f.create_dataset("uv", data=self.uv)
        f.create_dataset("nut", data=self.nut)
        f.create_dataset("yPlus",data=self.yPlus)
        f.create_dataset("uPlus", data=self.uPlus)
        f.create_dataset("uuPlus", data=self.uuPlus)
        f.create_dataset("vvPlus", data=self.vvPlus)
        f.create_dataset("wwPlus", data=self.wwPlus)
        f.create_dataset("uvPlus", data=self.uvPlus)
        f.create_dataset("kPlus", data=self.kPlus)
        f.create_dataset("uRms", data=self.uRms)
        f.create_dataset("vRms", data=self.vRms)
        f.create_dataset("wRms", data=self.wRms)

        f.close()

    def load(self, name):
        f = h5py.File(name, 'r')

        self.nu = f.attrs["nu"]
        self.uTau = f.attrs["uTau"]
        self.uB = f.attrs["uB"]
        self.uC = f.attrs["uC"]
        self.delta = f.attrs["delta"]
        self.delta99 = f.attrs["delta99"]
        self.deltaStar = f.attrs["deltaStar"]
        self.reTheta = f.attrs["theta"]
        self.reDelta99 = f.attrs["reDelta99"]
        self.reDeltaStar = f.attrs["reDeltaStar"]
        self.reTheta = f.attrs["reTheta"]
        self.reTau = f.attrs["reTau"]
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
        self.yPlus = f["yPlus"][:]
        self.uPlus = f["uPlus"][:]
        self.uuPlus= f["uuPlus"][:]
        self.vvPlus = f["vvPlus"][:]
        self.wwPlus = f["wwPlus"][:]
        self.uvPlus = f["uvPlus"][:]
        self.uvPlus = f["kPlus"][:]
        self.uRms = f["uRms"][:]
        self.vRms = f["vRms"][:]
        self.vRms = f["wRms"][:]
        self.kPlus = f["kPlus"][:]

        f.close()
