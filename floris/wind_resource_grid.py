import numpy as np

from floris.logging_manager import LoggingManager


class WindResourceGrid(LoggingManager):
    """
    WindResourceGrid class is used to read in hold the contents of wind resource grid (WRG)
    files.  The class provides methods for reading the files and extracting the data and
    converting to FLORIS data objects such as WindRose.

    The class is based on the floris_wrg repository by P.J. Stanley.

    Args:
        filename (str): The name of the WRG file to read.

    """

    def __init__(self, filename):
        self.filename = filename
        self.read_wrg_file(filename)

    def read_wrg_file(self, filename):
        # Read the file into data
        with open(filename, "r") as f:
            data = f.readlines()

        # Read the header
        header = data[0].split()
        self.nx = int(header[0])
        self.ny = int(header[1])
        self.xmin = float(header[2])
        self.ymin = float(header[3])
        self.grid_size = float(header[4])

        # The grid of points is implied by the values above
        self.x_array = np.arange(self.nx) * self.grid_size + self.xmin
        self.y_array = np.arange(self.ny) * self.grid_size + self.ymin

        # The number of grid points (n_gid) is the product of the number of points in x and y
        self.n_gid = self.nx * self.ny

        # Finally get the number of sectors from the first line after the header
        self.n_sectors = int(data[1][70:72])

        # The wind directions are implied by the number of sectors
        self.wind_directions = np.arange(0.0, 360.0, 360.0 / self.n_sectors)

        # Initialize the data arrays which have the same number of
        # elements as the number of grid points
        self.x = np.zeros(self.n_gid)
        self.y = np.zeros(self.n_gid)
        self.z = np.zeros(self.n_gid)
        self.h = np.zeros(self.n_gid)

        # Initialize the data arrays which are n_gid x n_sectors
        self.sector_freq = np.zeros((self.n_gid, self.n_sectors))
        self.weibull_A = np.zeros((self.n_gid, self.n_sectors))
        self.weibull_k = np.zeros((self.n_gid, self.n_sectors))

        # Loop through the data and extract the values
        for gid in range(self.n_gid):
            line = data[1 + gid]
            self.x[gid] = float(line[10:20])
            self.y[gid] = float(line[20:30])
            self.z[gid] = float(line[30:38])
            self.h[gid] = float(line[38:43])

            for sector in range(self.n_sectors):
                # The frequency of the wind in this sector is in probablility * 1000
                self.sector_freq[gid, sector] = (
                    float(line[72 + sector * 13 : 76 + sector * 13]) / 1000.0
                )

                # The A and k parameters are in the next 10 characters, with A stored * 10
                # and k stored * 100
                self.weibull_A[gid, sector] = (
                    float(line[76 + sector * 13 : 80 + sector * 13]) / 10.0
                )
                self.weibull_k[gid, sector] = (
                    float(line[80 + sector * 13 : 85 + sector * 13]) / 100.0
                )
