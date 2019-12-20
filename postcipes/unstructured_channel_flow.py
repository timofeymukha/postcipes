# This file is part of postcipes
# (c) Timofey Mukha
# The code is released under the MIT Licence.
# See LICENCE.txt and the Legal section in the README for more information

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from .postcipe import Postcipe
import numpy as np
from os.path import join
from scipy.integrate import simps
from collections import OrderedDict
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay
import h5py
import os
import turbulucid as tbl
import vtk
from vtk.numpy_interface import dataset_adapter as dsa
from vtk.util.numpy_support import *

__all__ = ["UnstructuredChannelFlow"]


class UnstructuredChannelFlow(Postcipe):

    def __init__(self, path, nu, n, time, wallModel=False):
        Postcipe.__init__(self)
        self.case = path
        self.readPath = join(self.case)
        self.nu = nu
        self.n = n
        self.wallModel = wallModel
        self.time = time

    def read(self, time, debug=False):
        """Read the case from a given path to .foam file.

        Parameters
        ----------
        time : float
            The time step to load, default to latest time

        Returns
        -------
            The reader updated with the read case.

        Raises
        ------
        ValueError
            If the path is not valid.


        """
        # Check that paths are valid

        if not os.path.exists(self.case):
            raise ValueError("Provided path to .foam file invalid!")

        if debug:
            print("    Opening the case")
        # Case reader
        reader = vtk.vtkOpenFOAMReader()
        reader.SetFileName(self.case)
        reader.Update()

        if debug:
            print("    Changing reader parameters")
        reader.CreateCellToPointOff()
        reader.DisableAllPointArrays()
        reader.EnableAllPatchArrays()
        reader.DecomposePolyhedraOn()
        reader.Update()
        reader.UpdateInformation()

        info = reader.GetExecutive().GetOutputInformation(0)

        if debug:
            print("The available timesteps are", vtk_to_numpy(reader.GetTimeValues()))

        if time is None:
            print("Selecting the latest available time step")
            info.Set(vtk.vtkStreamingDemandDrivenPipeline.UPDATE_TIME_STEP(),
                     vtk_to_numpy(reader.GetTimeValues())[-1])
        else:
            print("Selecting the time step", time)
            info.Set(vtk.vtkStreamingDemandDrivenPipeline.UPDATE_TIME_STEP(), time)

        reader.Update()
        reader.UpdateInformation()

        return reader

    def compute(self):
        d = 1 / self.n
        y = np.linspace(d / 2, 2 - d / 2, 2 * self.n)

        reader = self.read(self.time)
        caseData = reader.GetOutput()
        internalBlock = caseData.GetBlock(0)
        patchBlocks = caseData.GetBlock(1)

        bounds = internalBlock.GetBounds()

        fieldNames = dsa.WrapDataObject(internalBlock).GetCellData().keys()

        averaged = {}
        for i, field in enumerate(fieldNames):
            averaged[field] = []

        pointData = vtk.vtkCellDataToPointData()
        pointData.SetInputData(internalBlock)
        pointData.Update()

        plane = vtk.vtkPlaneSource()
        plane.SetResolution(int(bounds[1] / d), int(bounds[5] / d))

        kernel = vtk.vtkVoronoiKernel()

        interpolator = vtk.vtkPointInterpolator()
        interpolator.SetSourceData(pointData.GetOutput())
        interpolator.SetKernel(kernel)

        # Internal field, go layer by layer
        for i in range(y.size):
            plane.SetOrigin(0.55 * (bounds[0] + bounds[1]), y[i], 0.15 * (bounds[4] + bounds[5]))
            plane.SetPoint1(bounds[0], y[i], bounds[4])
            plane.SetPoint2(bounds[1], y[i], bounds[5])
            plane.Update()

            interpolator.SetInputConnection(plane.GetOutputPort())
            interpolator.Update()

            interpolatedData = dsa.WrapDataObject(interpolator.GetOutput()).GetPointData()
            for field in fieldNames:
                averaged[field].append(np.mean(interpolatedData[field], axis=0))

        # Patch data
        for wall in ["bottomWall", "topWall"]:
            wallBlock = patchBlocks.GetBlock(self.get_block_index(patchBlocks, wall))
            cellSizeFilter = vtk.vtkCellSizeFilter()
            cellSizeFilter.SetInputData(wallBlock)
            cellSizeFilter.Update()
            area = dsa.WrapDataObject(cellSizeFilter.GetOutput()).CellData['Area']

            wallData = dsa.WrapDataObject(wallBlock).CellData

            for field in fieldNames:
                # area weighted average
                avrg = np.sum(wallData[field] * area, axis=0) / np.sum(area)
                if wall == "bottomWall":
                    averaged[field].insert(0, avrg)
                else:
                    averaged[field].append(avrg)

        for field in fieldNames:
            averaged[field] = np.array(averaged[field])

        self.y = np.append(np.append(0, y), 2)
        self.avrgFields = averaged

        self.u = self.avrgFields['UMean'][:, 0]
        self.uu = self.avrgFields['UPrime2Mean'][:, 0]
        self.vv = self.avrgFields['UPrime2Mean'][:, 1]
        self.ww = self.avrgFields['UPrime2Mean'][:, 2]
        self.uv = self.avrgFields['UPrime2Mean'][:, 3]
        self.k = 0.5*(self.uu + self.vv + self.ww)
        self.nut = self.avrgFields['nutMean']

        self.tau = 0
        if self.wallModel:
            self.wss = self.avrgFields['wallShearStress'][:, 0]
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

    def get_block_index(self, blocks, name):
        """Get the index of the block by name.

        Parameters
        ----------
        blocks : vtkMultiBlockDataSet
            The dataset with the blocks.
        name : str
            The name of the block that is sought.

        Returns
        -------
        int
            The index of the sought block.

        """
        number = -1
        for i in range(blocks.GetNumberOfBlocks()):
            if (blocks.GetMetaData(i).Get(vtk.vtkCompositeDataSet.NAME()) ==
                    name):
                number = i
                break

        if number == -1:
            raise NameError("No block named " + name + " found")

        return number
