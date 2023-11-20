from nektsrs.io.helpers import read_int, read_real
import numpy as np
import struct

__all__ = ["NekInterpolatedData"]


class NekInterpolatedData:
    def __init__(
        self, points, stats, derivs, lx, start_time, end_time, avrg_time, nelems, bounds
    ):
        self.points = points
        self.stats = stats
        self.derivs = derivs
        self.lx = lx
        self.npoints = points.shape[0]
        self.nstats = stats.shape[1]
        self.nderivs = derivs.shape[1]
        self.start_time = start_time
        self.end_time = end_time
        self.avrg_time = avrg_time
        self.nelems = nelems
        self.bounds = bounds

    @classmethod
    def read(cls, fname: str):
        """Read data from interpolation file"""

        # open file
        infile = open(fname, "rb")
        # read header
        header = infile.read(460).split()

        # extract word size
        wdsize = int(header[1])

        # identify endian encoding
        etagb = infile.read(4)
        etag_l = struct.unpack("<f", etagb)[0]
        etag_l = int(etag_l * 1e5) / 1e5
        etag_b = struct.unpack(">f", etagb)[0]
        etag_b = int(etag_b * 1e5) / 1e5
        if etag_l == 6.54321:
            emode = "<"
        elif etag_b == 6.54321:
            emode = ">"
        else:
            raise ValueError("Could not determine endian")

        # get simulation parameters
        re_number = read_real(infile, emode, wdsize, 1)[0]
        bsize = read_real(infile, emode, wdsize, 3)
        belem = read_int(infile, emode, 3)
        pord = read_int(infile, emode, 3)
        nstat = read_int(infile, emode, 1)[0]
        nderiv = read_int(infile, emode, 1)[0]
        stime = read_real(infile, emode, wdsize, 4)
        nrec = read_int(infile, emode, 1)
        itime = read_real(infile, emode, wdsize, 1)
        npoints = read_int(infile, emode, 1)[0]

        # create main data structure
        ldim = 2

        points = np.zeros((npoints, ldim))

        # fill simulation parameters
        # self.bsize = bsize
        # self.belem = belem
        # self.pord = pord
        # self.start_time = stime[0]
        # self.end_time = stime[1]
        # self.effav_time = stime[2]
        # self.dt = stime[3]
        # self.nrec = nrec[0]
        # self.int_time = itime[0]

        # print(self.re_number, self.bsize, self.belem, self.pord)
        # print(self.start_time, self.end_time, self.effav_time, self.dt)

        # read coordinates
        for il in range(npoints):
            point = read_real(infile, emode, wdsize, ldim)
            points[il, 0] = point[0]
            points[il, 1] = point[1]

        stats = np.zeros((npoints, nstat))
        # read statistics
        for il in range(nstat):
            for jl in range(npoints):
                value = read_real(infile, emode, wdsize, 1)
                stats[jl, il] = value[0]

        derivs = np.zeros((npoints, nderiv))
        # read derivatives
        for il in range(nderiv):
            for jl in range(npoints):
                value = read_real(infile, emode, wdsize, 1)
                derivs[jl, il] = value[0]

        return cls(
            points,
            stats,
            derivs,
            np.max(pord),
            stime[0],
            stime[1],
            stime[2],
            belem,
            bsize,
        )

    def print(self):
        print("Start time", self.start_time)
        print("End time", self.end_time)
        print("Avrg time", self.avrg_time)
        print("N points", self.npoints)
        print("Lx", self.lx)
        print("Bounds", self.bounds)
        print("N elements", self.nelems)
