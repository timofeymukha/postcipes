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
from scipy.integrate import simps
from scipy.interpolate import interp1d

__all__ = ["ChannelFlow"]


class ChannelFlow(Postcipe):

    def __init__(self, path=None, nu=None, time=None, wallModel=False, kBudget=False, readH=False, initEmpty=False,
                 heat=False):
        Postcipe.__init__(self)
        if initEmpty:
            return
        self.case = path
        self.readPath = join(self.case, "postProcessing", "collapsedFields",
                             str(time))
        self.nu = nu
        self.time = time

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

        if kBudget:
            self.kProduction = np.genfromtxt(join(self.readPath, "kProduction.xy"))[:, 1]
            self.kConvection = np.genfromtxt(join(self.readPath, "kConvection.xy"))[:, 1]
            self.kViscousDiffusion = np.genfromtxt(join(self.readPath, "kViscousDiffusion.xy"))[:, 1]
            self.kPressureVelocityTransport = np.genfromtxt(join(self.readPath, "kPressureVelocityTransportMean.xy"))[:, 1]
            self.kTurbulentTransport = np.genfromtxt(join(self.readPath, "kTurbulentTransportMean.xy"))[:, 1]
            self.kDissipation = np.genfromtxt(join(self.readPath, "kDissipationMean.xy"))[:, 1]
            self.kSgsDiffusion = np.genfromtxt(join(self.readPath, "kSgsDiffusion.xy"))[:, 1]

            self.kBalance = np.copy(self.kProduction)
            self.kBalance += self.kConvection
            self.kBalance += self.kViscousDiffusion
            self.kBalance += self.kPressureVelocityTransport
            self.kBalance += self.kTurbulentTransport
            self.kBalance += self.kDissipation

        if heat:
            self.t = np.genfromtxt(join(self.readPath, "TMean.xy"))[:, 1]
            self.tt = np.genfromtxt(join(self.readPath, "TPrime2Mean.xy"))[:, 1]
            self.tu = np.genfromtxt(join(self.readPath, "TUMean_X.xy"))[:, 1] - self.t*self.u
            self.tv = np.genfromtxt(join(self.readPath, "TUMean_Y.xy"))[:, 1] - self.t*self.v
            self.tw = np.genfromtxt(join(self.readPath, "TUMean_Z.xy"))[:, 1] - self.t*self.w

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

    def utau_relative_error(self, bench, procent=True, abs=False):
        error = (self.uTau - bench)/bench
        if procent:
            error *= 100
        if abs:
            error = np.abs(error)

        return error

    def cell_average_dns(self, dnsY, dnsV):
        dnsInterp = interp1d(np.append(dnsY, [1]), np.append(dnsV, dnsV[-1]))
        yf = self.yf[self.yf < self.delta + 1e-10]
        print(yf)

        dnsAvrg = np.zeros(yf.size - 1)

        for i, _ in enumerate(dnsAvrg):
            yCell = dnsY[np.logical_and(dnsY > yf[i], dnsY < yf[i + 1])]
            vCell = dnsV[np.logical_and(dnsY > yf[i], dnsY < yf[i + 1])]

            yCell = np.append(np.append(yf[i], yCell), yf[i + 1])
            vStart = dnsInterp(yf[i])
            vEnd = dnsInterp(yf[i + 1])
            vCell = np.append(np.append(vStart, vCell), vEnd)

            print(yCell)
            dnsAvrg[i] = simps(vCell, x=yCell)/(yCell[-1] - yCell[0])

        return dnsAvrg




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
