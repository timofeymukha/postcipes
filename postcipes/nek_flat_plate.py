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
import pymech.dataset as ds
from os.path import join
from scipy.interpolate import interp1d
import glob
from scipy.special import roots_jacobi

__all__ = ["NekFlatPlate"]


class NekFlatPlate(Postcipe):

    def __init__(self, path, basename, starttime=0, lx1=8, nu=None,
                 nutstats=False):
        Postcipe.__init__(self)

        self.case = path
        self.basename = basename
        self.nu = nu
        self.lx1 = lx1
        self.nutstats = nutstats

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

        keys = ["s01", "s02", "s03", "s04", "s05", "s06", "s07", "s09", "s11", "s67", "s68", "s69"]

        if nutstats is True:
            keys += ["s45", "s46", "s47", "s48", "s49", "s50", "s51", "s52",
                     "s53", "s54"]

        average_data = {}
        for key in keys:
            average_data[key] = np.zeros((datasets[0].x.size, datasets[0].y.size))

        print("Averaging in time")
        for i, d in enumerate(datasets):
            for key in keys:
                average_data[key] += weights[i]*np.transpose(np.array(d[key][0, :, :]))

        self.u = average_data["s01"]
        self.v = average_data["s02"]
        self.w = average_data["s03"]
        self.p = average_data["s04"]
        self.uu = average_data["s05"] - self.u*self.u
        self.vv = average_data["s06"] - self.v*self.v
        self.ww = average_data["s07"] - self.w*self.w
        self.uv = average_data["s09"] - self.u*self.v
        self.uw = average_data["s11"]

        self.taux = average_data["s67"]
        self.tauy = average_data["s68"]
        self.tauz = average_data["s69"]

        if nutstats:
            self.nutotdudx = average_data["s45"]
            self.nutotdudy = average_data["s46"]
            self.nutotdudz = average_data["s47"]
            self.nutotdvdx = average_data["s48"]
            self.nutotdvdy = average_data["s49"]
            self.nutotdvdz = average_data["s50"]
            self.nutotdwdx = average_data["s51"]
            self.nutotdwdy = average_data["s52"]
            self.nutotdwdz = average_data["s53"]
            self.nutot = average_data["s54"]

        self.x = datasets[0].x.data
        self.y = datasets[0].y.data

    def save(self, name):
        f = h5py.File(name, 'w')

        f.attrs["nu"] = self.nu
        f.attrs["lx1"] = self.lx1

        f.create_dataset("x", data=self.x)
        f.create_dataset("y", data=self.y)
        f.create_dataset("u", data=self.u)
        f.create_dataset("v", data=self.v)
        f.create_dataset("p", data=self.p)
        f.create_dataset("uu", data=self.uu)
        f.create_dataset("vv", data=self.vv)
        f.create_dataset("ww", data=self.ww)
        f.create_dataset("uv", data=self.uv)
        f.create_dataset("uw", data=self.uw)
        f.create_dataset("taux", data=self.taux)
        f.create_dataset("tauy", data=self.tauy)
        f.create_dataset("tauz", data=self.tauz)

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
        f.close()
