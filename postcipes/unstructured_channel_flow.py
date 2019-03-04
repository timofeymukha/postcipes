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
from collections import OrderedDict
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay
import h5py

__all__ = ["UnstructuredChannelFlow"]


class UnstructuredChannelFlow(Postcipe):

    def __init__(self, path, nu, nSamples, wallModel=False):
        Postcipe.__init__(self)
        self.case = path
        self.readPath = join(self.case)
        self.nu = nu

        self.tblCase = tbl.Case(self.readPath)
        self.nSamples = nSamples
        self.wallModel = wallModel

        # line = vtk.vtkLineSource()
        # probeFilter = vtk.vtkProbeFilter()
        # probeFilter.SetSourceData(self.tblCase.vtkData.VTKObject)


        # smallDx = 9/(2*nSamples)
        # for seed in range(int(nSeedPoints)):
        #
        #     seedPoint = seeds[seed]
        #     line.SetResolution(nSamples - 1)
        #     line.SetPoint1(0 + smallDx, seedPoint, 0)
        #     line.SetPoint2(9 - smallDx, seedPoint, 0)
        #     line.Update()
        #
        #     probeFilter.SetInputConnection(line.GetOutputPort())
        #     probeFilter.Update()
        #
        #     probeData = dsa.WrapDataObject(probeFilter.GetOutput()).PointData
        #
        #     for field in avrgFields:
        #         if avrgFields[field].shape[1] == 9:  # a tensor
        #             reshaped = probeData[field].reshape((nSamples, 9))
        #             avrgFields[field][seed] = np.mean(reshaped, axis=0)
        #         else:
        #             avrgFields[field][seed] = np.mean(probeData[field], axis=0)
        #
        # self.avrgFields = avrgFields

    def compute(self):
        seeds = np.sort(self.tblCase.boundary_data("inlet")[0][:, 1])
        avrgFields = OrderedDict()

        cellData = self.tblCase.vtkData.GetCellData()
        nFields = cellData.GetNumberOfArrays()
        nSeedPoints = seeds.size

        for field in range(nFields):
            name = cellData.GetArrayName(field)
            nCols = cellData.GetArray(field).GetNumberOfComponents()
            avrgFields[name] = np.zeros((nSeedPoints, nCols))

        coords = np.row_stack((self.tblCase.cellCentres,
                               self.tblCase.boundary_data("inlet")[0],
                               self.tblCase.boundary_data("outlet")[0]))
        delaunay = Delaunay(coords)

        dx = 9/self.nSamples
        for field in avrgFields:
            if np.ndim(self.tblCase[field]) == 1:
                data = np.row_stack((self.tblCase[field][:, np.newaxis],
                                     self.tblCase.boundary_data("inlet")[1][field][:, np.newaxis],
                                     self.tblCase.boundary_data("outlet")[1][field][:, np.newaxis]))
            else:
                data = np.row_stack((self.tblCase[field],
                                     self.tblCase.boundary_data("inlet")[1][field],
                                     self.tblCase.boundary_data("outlet")[1][field]))

            interpolant = LinearNDInterpolator(delaunay, data)
            for seed in range(int(nSeedPoints)):
                x = dx/2
                for i in range(self.nSamples-1):
                    avrgFields[field][seed] += interpolant([x, seeds[seed]])[0]
                    x += dx
                avrgFields[field][seed] /= (self.nSamples-1)

        self.avrgFields = avrgFields

        self.y = np.append(np.append([0], seeds), [2])
        bot = np.mean(self.tblCase.boundary_data("bottomWall")[1]['UMean'][:,0])
        top = np.mean(self.tblCase.boundary_data("topWall")[1]['UMean'][:,0])
        self.u = np.append(np.append(bot, avrgFields['UMean'][:, 0]), top)
        bot = np.mean(self.tblCase.boundary_data("bottomWall")[1]['UPrime2Mean'][:,0])
        top = np.mean(self.tblCase.boundary_data("topWall")[1]['UPrime2Mean'][:,0])
        self.uu = np.append(np.append(bot, avrgFields['UPrime2Mean'][:, 0]), top)
        bot = np.mean(self.tblCase.boundary_data("bottomWall")[1]['UPrime2Mean'][:,1])
        top = np.mean(self.tblCase.boundary_data("topWall")[1]['UPrime2Mean'][:,1])
        self.vv = np.append(np.append(bot, avrgFields['UPrime2Mean'][:, 1]), top)
        bot = np.mean(self.tblCase.boundary_data("bottomWall")[1]['UPrime2Mean'][:,2])
        top = np.mean(self.tblCase.boundary_data("topWall")[1]['UPrime2Mean'][:,2])
        self.ww = np.append(np.append(bot, avrgFields['UPrime2Mean'][:, 2]), top)
        bot = np.mean(self.tblCase.boundary_data("bottomWall")[1]['UPrime2Mean'][:,3])
        top = np.mean(self.tblCase.boundary_data("topWall")[1]['UPrime2Mean'][:,3])
        self.uv = np.append(np.append(bot, avrgFields['UPrime2Mean'][:, 3]), top)
        self.k = 0.5*(self.uu + self.vv + self.ww)
        bot = np.mean(self.tblCase.boundary_data("bottomWall")[1]['nutMean'])
        top = np.mean(self.tblCase.boundary_data("topWall")[1]['nutMean'])
        self.nut = np.append(np.append(bot, avrgFields['nutMean']), top)
        bot = np.mean(self.tblCase.boundary_data("bottomWall")[1]['wallShearStressMean'][:, 0])
        top = np.mean(self.tblCase.boundary_data("topWall")[1]['wallShearStressMean'][:, 0])

        self.tau = 0
        if self.wallModel:
            self.wss = np.append(np.append(bot, avrgFields['wallShearStress'][:, 0]), top)
            self.tau = 0.5*(self.wss[0] + self.wss[-1])
        else:
            self.tau = self.nu*0.5*(self.u[1] + self.u[-2])/self.y[1]

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
        #
        self.deltaStar = tbl.delta_star(self.y[:int(self.y.size/2)],
                                        self.u[:int(self.u.size/2)],
                                        interpolate=True)
        #
        self.reTheta = self.theta*self.uC/self.nu
        self.reDelta99 = self.delta99*self.uC/self.nu
        self.reDeltaStar = self.deltaStar*self.uC/self.nu


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

    def utau_relative_error(self, bench, procent=True, abs=False):
        error = (self.uTau - bench)/bench
        if procent:
            error *= 100
        if abs:
            error = np.abs(error)

        return error
