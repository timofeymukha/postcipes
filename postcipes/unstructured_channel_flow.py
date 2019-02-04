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
import vtk
from collections import OrderedDict
from vtk.numpy_interface import dataset_adapter as dsa

__all__ = ["UnstructuredChannelFlow"]


class UnstructuredChannelFlow(Postcipe):

    def __init__(self, path, nu, nSamples, wallModel=False):
        Postcipe.__init__(self)
        self.case = path
        self.readPath = join(self.case, "averaged.vtm")
        self.nu = nu

        self.tblCase = tbl.Case(self.readPath)
        seeds = self.tblCase.boundary_data("inlet")[0][:, 1]

        line = vtk.vtkLineSource()
        probeFilter = vtk.vtkProbeFilter()
        probeFilter.SetSourceData(self.tblCase.vtkData.VTKObject)

        avrgFields = OrderedDict()

        cellData = self.tblCase.vtkData.GetCellData()
        nFields = cellData.GetNumberOfArrays()
        nSeedPoints = seeds.size

        for field in range(nFields):
            name = cellData.GetArrayName(field)
            nCols = cellData.GetArray(field).GetNumberOfComponents()
            avrgFields[name] = np.zeros((nSeedPoints, nCols))


        smallDx = 9/(2*nSamples)
        for seed in range(int(nSeedPoints)):

            seedPoint = seeds[seed]
            line.SetResolution(nSamples - 1)
            line.SetPoint1(0 + smallDx, seedPoint, 0)
            line.SetPoint2(0 - smallDx, seedPoint, 0)
            line.Update()

            probeFilter.SetInputConnection(line.GetOutputPort())
            probeFilter.Update()

            probeData = dsa.WrapDataObject(probeFilter.GetOutput()).PointData

            for field in avrgFields:
                if avrgFields[field].shape[1] == 9:  # a tensor
                    reshaped = probeData[field].reshape((nSamples, 9))
                    avrgFields[field][seed] = np.mean(reshaped, axis=0)
                else:
                    avrgFields[field][seed] = np.mean(probeData[field], axis=0)

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
        self.wss = np.append(np.append(bot, avrgFields['wallShearStress'][:, 0]), top)

        self.tau = 0
        if wallModel:
            self.tau = 0.5*(self.wss[0] + self.wss[-1])
        else:
            self.tau = nu*0.5*(self.u[1] + self.u[-2])/self.y[1]

        self.uTau = np.sqrt(self.tau)
        self.delta = 0.5*(self.y[-1] - self.y[0])
        self.uB = simps(self.u, self.y)/(self.delta)
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

    def utau_relative_error(self, bench, procent=True, abs=False):
        error = (self.uTau - bench)/bench
        if procent:
            error *= 100
        if abs:
            error = np.abs(error)

        return error
