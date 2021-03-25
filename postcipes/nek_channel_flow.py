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

__all__ = ["NekChannelFlow", "gll", "gll_int", "lagrange_interpolate"]


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


def lagrange_interpolate(y, coeffs, lx1, ppelem=None):
    nely = y.size / lx1
    offset = 0

    if ppelem is None:
        ppelem = 10 * lx1

    totaly = []
    vals = []
    for e in range(int(nely)):
        ind = np.arange(offset, offset + lx1)
        ylocal = np.linspace(y[ind[0]], y[ind[-1]], ppelem)
        totaly.append(ylocal)
        vals.append(np.poly1d(coeffs[e])(ylocal))
        offset += lx1
    return np.array(totaly).flatten(), np.array(vals).flatten()


class NekChannelFlow(Postcipe):

    def __init__(self, path, basename, starttime=0, lx1=8, nu=None,
                 nutstats=False):
        Postcipe.__init__(self)

        self.case = path
        self.basename = basename
        self.nu = nu
        self.lx1 = lx1
        self.nutstats = nutstats
        datafiles = glob.glob(self.case + '\\sts' + basename +'[0-1].f[0-9][0-9][0-9][0-9][0-9]')
        datasets = [ds.open_dataset(i) for i in datafiles]

        if len(datasets) is 0:
            raise FileExistsError

        namemap = {"s01": "u", "s02": "v", "s03" : "w", "s05": "uu",
                   "s06": "vv", "s07": "ww", "s09": "uv"}

        ndatasets = len(datasets)
        nx = datasets[0].x.size
        nelx = nx / lx1

        keys = ["s01", "s02", "s03", "s05", "s06", "s07", "s09"]
        if nutstats is True:
            keys = keys + ["s45", "s46", "s47", "s48", "s49", "s50", "s51", "s52",
                        "s53", "s54"]

        integral = {}
        for key in keys:
            integral[key] = np.zeros(datasets[0].y.size)

        for d in datasets:
            if d.time.data < starttime:
                ndatasets -= 1
                continue
            for key in keys:
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
        self.v = integral["s02"]
        self.w = integral["s03"]
        self.uu = integral["s05"] - self.u*self.u
        self.vv = integral["s06"]
        self.ww = integral["s07"]
        self.uv = integral["s09"]

        if nutstats:
            self.nutotdudx = integral["s45"]
            self.nutotdudy = integral["s46"]
            self.nutotdudz = integral["s47"]
            self.nutotdvdx = integral["s48"]
            self.nutotdvdy = integral["s49"]
            self.nutotdvdz = integral["s50"]
            self.nutotdwdx = integral["s51"]
            self.nutotdwdy = integral["s52"]
            self.nutotdwdz = integral["s53"]
            self.nutot = integral["s54"]

        self.y = datasets[0].y.data
        self.compute_interpolants()
        dudyb = np.poly1d(self.u_polys[0]).deriv()(self.y[0])
        dudyt = np.abs(np.poly1d(self.u_polys[-1]).deriv()(self.y[-1]))
        dudy = 0.5 * (dudyb + dudyt)
        self.tau_w = nu*dudy
        self.utau = np.sqrt(self.tau_w)
        self.retau = self.utau / nu * 0.5*(self.y[-1] - self.y[0])

        self.yplus = self.y*self.utau/self.nu

    def compute_interpolants(self):
        from scipy.interpolate import lagrange
        lx1 = self.lx1

        nely = int(self.y.size/lx1)
        offset = 0

        u = []
        v = []
        w = []
        uu = []
        vv = []
        ww = []
        uv = []
        for e in range(nely):
            ind = np.arange(offset, offset + lx1)
            p = lagrange(self.y[ind], self.u[ind]).coeffs
            u.append(p)
            p = lagrange(self.y[ind], self.v[ind]).coeffs
            v.append(p)
            p = lagrange(self.y[ind], self.w[ind]).coeffs
            w.append(p)
            p = lagrange(self.y[ind], self.uu[ind]).coeffs
            uu.append(p)
            p = lagrange(self.y[ind], self.vv[ind]).coeffs
            vv.append(p)
            p = lagrange(self.y[ind], self.ww[ind]).coeffs
            ww.append(p)
            p = lagrange(self.y[ind], self.uv[ind]).coeffs
            uv.append(p)
            offset += lx1

        self.u_polys = np.array(u)
        self.v_polys = np.array(v)
        self.w_polys = np.array(w)
        self.uu_polys = np.array(uu)
        self.vv_polys = np.array(vv)
        self.ww_polys = np.array(ww)
        self.uv_polys = np.array(uv)


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
        f.attrs["lx1"] = self.lx1
        f.attrs["utau"] = self.utau
#        f.attrs["uB"] = self.uB
#        f.attrs["uC"] = self.uC
#        f.attrs["delta"] = self.delta
#        f.attrs["delta99"] = self.delta99
#        f.attrs["deltaStar"] = self.deltaStar
#        f.attrs["theta"] = self.reTheta
#        f.attrs["reDelta99"] = self.reDelta99
#        f.attrs["reDeltaStar"] = self.reDeltaStar
#        f.attrs["reTheta"] = self.reTheta
        f.attrs["retau"] = self.retau
#        f.attrs["reB"] = self.reB
#        f.attrs["reC"] = self.reC

        f.create_dataset("y", data=self.y)
        f.create_dataset("u", data=self.u)
        f.create_dataset("uu", data=self.uu)
        f.create_dataset("vv", data=self.vv)
        f.create_dataset("ww", data=self.ww)
#        f.create_dataset("k", data=self.k)
        f.create_dataset("uv", data=self.uv)
#        f.create_dataset("nut", data=self.nut)
        f.create_dataset("yplus",data=self.yplus)

        if self.nutstats:
            f.create_dataset("nutotdudx", data=self.nutotdudx)
            f.create_dataset("nutotdudy", data=self.nutotdudy)
            f.create_dataset("nutotdudz", data=self. nutotdudz)
            f.create_dataset("nutotdvdx", data=self. nutotdvdx)
            f.create_dataset("nutotdvdy", data=self.nutotdvdy)
            f.create_dataset("nutotdvdz", data=self.nutotdvdz)
            f.create_dataset("nutotdwdx", data=self.nutotdwdx)
            f.create_dataset("nutotdwdy", data=self.nutotdwdy)
            f.create_dataset("nutotdwdz", data=self.nutotdwdz)
            f.create_dataset("nutot",     data=self.nutot)



#        f.create_dataset("uPlus", data=self.uPlus)
#        f.create_dataset("uuPlus", data=self.uuPlus)
#        f.create_dataset("vvPlus", data=self.vvPlus)
#        f.create_dataset("wwPlus", data=self.wwPlus)
#        f.create_dataset("uvPlus", data=self.uvPlus)
#        f.create_dataset("kPlus", data=self.kPlus)
#        f.create_dataset("uRms", data=self.uRms)
#        f.create_dataset("vRms", data=self.vRms)
#        f.create_dataset("wRms", data=self.wRms)

        f.create_dataset("u_polys", data=self.u_polys)
        f.create_dataset("v_polys", data=self.v_polys)
        f.create_dataset("w_polys", data=self.w_polys)
        f.create_dataset("uu_polys", data=self.uu_polys)
        f.create_dataset("vv_polys", data=self.vv_polys)
        f.create_dataset("ww_polys", data=self.ww_polys)
        f.create_dataset("uv_polys", data=self.uv_polys)

        f.close()
