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
                 nutstats=False, heat=False):
        Postcipe.__init__(self)

        self.case = path
        self.basename = basename
        self.nu = nu
        self.lx1 = lx1
        self.nutstats = nutstats
        self.heat = heat
        
        
        datafiles = glob.glob(join(self.case, 'sts' + basename +
                                   '[0-1].f[0-9][0-9][0-9][0-9][0-9]'))
        print("Reading datasets")
        datasets = [ds.open_dataset(i) for i in datafiles]

        print("Filtering by write time")
        filtered_times = []
        kept_times = []
        new_datasets = []
        for d in datasets:
            if d.time.data < starttime:
                filtered_times.append(d.time.data)
                continue
            else:
                new_datasets.append(d)
                kept_times.append(d.time.data)
        datasets = new_datasets

        if len(datasets) == 0:
            raise FileExistsError

        # NB: datasets may not be in order of time, so we resort the datasets
        filtered_times = np.array(filtered_times)
        kept_times = np.array(kept_times)
    
        time_ind = kept_times.argsort()
        kept_times.sort()
        filtered_times.sort()
        new_datasets = []
        for i in range(len(datasets)):
            new_datasets.append(datasets[time_ind[i]])
        datasets = new_datasets
        
        print("A total of " + str(len(kept_times)) + " are kept")
        print("Kept times", kept_times)
        print("Filtered times", filtered_times)
        dt = kept_times[1:] - kept_times[0:-1]

        # Take care of the length of the first dataset
        if len(filtered_times) == 0:
            print("No datasets are filtered")
            print("Assuming the first dataset is as long as the second")
            dt = np.insert(dt, 0, dt[0])
            total_time = np.sum(dt)
        else:
            print(str(len(filtered_times)) + " datasets are filtered")
            dt = np.insert(dt, 0, kept_times[0] - filtered_times[-1])
            total_time = np.sum(dt)
        
        weights = dt/total_time

        print("Weights", weights)
        print("Total averaging time:", total_time)
        print("Dataset lengths", dt)
        

        nx = datasets[0].x.size
        nelx = nx / lx1

        keys = ["s01", "s02", "s03", "s05", "s06", "s07", "s09", "s10", "s11"]
        if nutstats is True:
            keys += ["s45", "s46", "s47", "s48", "s49", "s50", "s51", "s52",
                     "s53", "s54"]

        if heat is True:
            keys += ["s55", "s56", "s57", "s58", "s59", "s60", "s61", "s62"]

        if nutstats and heat:
            keys += ["s63", "s64", "s65", "s66"]

        integral = {}
        for key in keys:
            integral[key] = np.zeros(datasets[0].y.size)

        
        print("Averaging in time")
        for i, d in enumerate(datasets):
            for key in keys:
                if i == 0:
                    datasets[0][key] = weights[i]*d[key]
                else:
                    datasets[0][key] += weights[i]*d[key]

        self.data2d = datasets[0]

        print("Averaging in space")
        for key in keys:
            offset = 0
            for _ in range(int(nelx)):
                ind = np.arange(offset, offset + lx1)
                a = datasets[0].x[offset]
                b = datasets[0].x[offset + lx1 - 1]
                for j in range(d.y.size):
                    slice = datasets[0][key].isel(x=ind, y=j).data[0]
                    integral[key][j] += gll_int(slice, a, b)

                offset += lx1

        for key in integral:
            integral[key] /= float(datasets[0]["x"].max() - datasets[0]["x"].min())

        self.u = integral["s01"]
        self.v = integral["s02"]
        self.w = integral["s03"]
        self.uu = integral["s05"] - self.u*self.u
        self.vv = integral["s06"] - self.v*self.v
        self.ww = integral["s07"] - self.w*self.w
        self.uv = integral["s09"] - self.u*self.v
        self.uw = integral["s11"] - self.u*self.w
        self.vw = integral["s10"] - self.v*self.w

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

        if heat:
            self.t = integral["s55"]
            self.tt = integral["s56"] - self.t*self.t
            self.tu = integral["s57"] - self.t*self.u
            self.tv = integral["s58"] - self.t*self.v
            self.tw = integral["s59"] - self.t*self.w
            self.dtdx = integral["s60"]
            self.dtdy = integral["s61"]
            self.dtdz = integral["s62"]

        if heat and nutstats:
            self.xitotdtdx = integral["s63"]
            self.xitotdtdy = integral["s64"]
            self.xitotdtdz = integral["s65"]
            self.xitot = integral["s66"]


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
        uw = []
        vw = []
        nutotdudy = []
        nutot = []

        for _ in range(nely):
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
            p = lagrange(self.y[ind], self.uw[ind]).coeffs
            uw.append(p)
            p = lagrange(self.y[ind], self.vw[ind]).coeffs
            vw.append(p)
            p = lagrange(self.y[ind], self.nutotdudy[ind]).coeffs
            nutotdudy.append(p)
            p = lagrange(self.y[ind], self.nutot[ind]).coeffs
            nutot.append(p)
            offset += lx1

        self.u_polys = np.array(u)
        self.v_polys = np.array(v)
        self.w_polys = np.array(w)
        self.uu_polys = np.array(uu)
        self.vv_polys = np.array(vv)
        self.ww_polys = np.array(ww)
        self.uv_polys = np.array(uv)
        self.uw_polys = np.array(uw)
        self.vw_polys = np.array(vw)
        self.nutotdudy_polys = np.array(nutotdudy)
#        self.nutot_polys = np.array(nutot)

    def save(self, name):
        f = h5py.File(name, 'w')

        f.attrs["nu"] = self.nu
        f.attrs["lx1"] = self.lx1
        f.attrs["utau"] = self.utau
        f.attrs["retau"] = self.retau

        f.create_dataset("y", data=self.y)
        f.create_dataset("u", data=self.u)
        f.create_dataset("v", data=self.v)
        f.create_dataset("w", data=self.w)
        f.create_dataset("uu", data=self.uu)
        f.create_dataset("vv", data=self.vv)
        f.create_dataset("ww", data=self.ww)
#        f.create_dataset("k", data=self.k)
        f.create_dataset("uv", data=self.uv)
        f.create_dataset("uw", data=self.uw)
        f.create_dataset("vw", data=self.vw)
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
#            f.create_dataset("nutot_polys", data=self.nutot_polys)
            f.create_dataset("nutotdudy_polys", data=self.nutotdudy_polys)
            
        if self.heat:
            f.create_dataset("t", data=self.t)
            f.create_dataset("tt", data=self.tt)
            f.create_dataset("tu", data=self.tu)
            f.create_dataset("tv", data=self.tv)
            f.create_dataset("tw", data=self.tw)
            f.create_dataset("dtdx", data=self.dtdx)
            f.create_dataset("dtdy", data=self.dtdy)
            f.create_dataset("dtdz", data=self.dtdz)

            if self.nutstats:
                f.create_dataset("xitotdtdx", data=self.xitotdtdx)
                f.create_dataset("xitotdtdy", data=self.xitotdtdy)
                f.create_dataset("xitotdtdz", data=self.xitotdtdz)
                f.create_dataset("xitot", data=self.xitot)
                
            

        f.create_dataset("u_polys", data=self.u_polys)
        f.create_dataset("v_polys", data=self.v_polys)
        f.create_dataset("w_polys", data=self.w_polys)
        f.create_dataset("uu_polys", data=self.uu_polys)
        f.create_dataset("vv_polys", data=self.vv_polys)
        f.create_dataset("ww_polys", data=self.ww_polys)
        f.create_dataset("uv_polys", data=self.uv_polys)
        f.create_dataset("uw_polys", data=self.uw_polys)
        f.create_dataset("vw_polys", data=self.vw_polys)

        f.close()
