import numpy as np
from scipy.interpolate import RegularGridInterpolator

from floris import WindRose, WindRoseByTurbine
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
        """
        Read the contents of a WRG file and store the data in the object.

        Args:
            filename (str): The name of the WRG file to read.

        """

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
        x_gid = np.zeros(self.n_gid)
        y_gid = np.zeros(self.n_gid)
        z_gid = np.zeros(self.n_gid)
        h_gid = np.zeros(self.n_gid)

        # Initialize the data arrays which are n_gid x n_sectors
        sector_freq_gid = np.zeros((self.n_gid, self.n_sectors))
        weibull_A_gid = np.zeros((self.n_gid, self.n_sectors))
        weibull_k_gid = np.zeros((self.n_gid, self.n_sectors))

        # Loop through the data and extract the values
        for gid in range(self.n_gid):
            line = data[1 + gid]
            x_gid[gid] = float(line[10:20])
            y_gid[gid] = float(line[20:30])
            z_gid[gid] = float(line[30:38])
            h_gid[gid] = float(line[38:43])

            for sector in range(self.n_sectors):
                # The frequency of the wind in this sector is in probablility * 1000
                sector_freq_gid[gid, sector] = (
                    float(line[72 + sector * 13 : 76 + sector * 13]) / 1000.0
                )

                # The A and k parameters are in the next 10 characters, with A stored * 10
                # and k stored * 100
                weibull_A_gid[gid, sector] = float(line[76 + sector * 13 : 80 + sector * 13]) / 10.0
                weibull_k_gid[gid, sector] = (
                    float(line[80 + sector * 13 : 85 + sector * 13]) / 100.0
                )

        # Save a single value of z and h for the entire grid
        self.z = z_gid[0]
        self.h = h_gid[0]

        # Index the by sector data by x and y
        self.sector_freq = np.zeros((self.nx, self.ny, self.n_sectors))
        self.weibull_A = np.zeros((self.nx, self.ny, self.n_sectors))
        self.weibull_k = np.zeros((self.nx, self.ny, self.n_sectors))

        for x_idx, x in enumerate(self.x_array):
            for y_idx, y in enumerate(self.y_array):
                # Find the indices when x_gid and y_gid are equal to x and y
                idx = np.where((x_gid == x) & (y_gid == y))[0]

                # Assign the data to the correct location
                self.sector_freq[x_idx, y_idx, :] = sector_freq_gid[idx, :]
                self.weibull_A[x_idx, y_idx, :] = weibull_A_gid[idx, :]
                self.weibull_k[x_idx, y_idx, :] = weibull_k_gid[idx, :]

        # Build the interpolant function lists
        self.interpolant_sector_freq = self._build_interpolant_function_list(
            self.x_array, self.y_array, self.n_sectors, self.sector_freq
        )
        self.interpolant_weibull_A = self._build_interpolant_function_list(
            self.x_array, self.y_array, self.n_sectors, self.weibull_A
        )
        self.interpolant_weibull_k = self._build_interpolant_function_list(
            self.x_array, self.y_array, self.n_sectors, self.weibull_k
        )

    def __str__(self) -> str:
        """
        Return a string representation of the WindRose object
        """

        return (
            f"WindResourceGrid with {self.nx} x {self.ny} grid points, "
            f"min x: {self.xmin}, min y: {self.ymin}, grid size: {self.grid_size}, "
            f"z: {self.z}, h: {self.h}, {self.n_sectors} sectors"
        )

    def _build_interpolant_function_list(self, x, y, n_sectors, data):
        """
        Build a list of interpolant functions for the data.  It is assumed that the function
        should return a list of interpolant functions, length n_sectors.

        Args:
            x (np.array): The x values of the data, length nx.
            y (np.array): The y values of the data, length ny.
            n_sectors (int): The number of sectors.
            data (np.array): The data to interpolate, shape (nx, ny, n_sectors).

        Returns:
            list: A list of interpolant functions, length n_sectors.
        """

        function_list = []

        for sector in range(n_sectors):
            function_list.append(
                RegularGridInterpolator(
                    (x, y),
                    data[:, :, sector],
                    bounds_error=False,
                    fill_value=None,
                )
            )

        return function_list

    def _interpolate_data(self, x, y, interpolant_function_list):
        """
        Interpolate the data at a given x, y location using the interpolant function list.

        Args:
            x (float): The x location to interpolate.
            y (float): The y location to interpolate.
            interpolant_function_list (list): A list of interpolant functions.

        Returns:
            list: A list of interpolated data, length n_sectors.
        """

        # Check if x and y are within the bounds of the self.x_array and self.y_array, if
        # so use the nearest method, otherwise use the linear method of interpolation
        if (
            x < self.x_array[0]
            or x > self.x_array[-1]
            or y < self.y_array[0]
            or y > self.y_array[-1]
        ):
            method = "nearest"
        else:
            method = "linear"

        result = np.zeros(self.n_sectors)
        for sector in range(self.n_sectors):
            result[sector] = interpolant_function_list[sector]((x, y), method=method)

        return result

    def _weibull_cumulative(self, x, a, k):
        """
        Calculate the Weibull cumulative distribution function.
        """

        exponent = -((x / a) ** k)
        result = 1.0 - np.exp(exponent)

        # Where x is less than 0, the result should be 0
        result[x < 0] = 0.0

        return result

        # Original code from PJ Stanley
        # if x >= 0.0:
        #     exponent = -(x / a) ** k
        #     return 1.0 - np.exp(exponent)
        # else:
        #     return 0.0

    def _generate_wind_speed_frequencies_from_weibull(self, A, k, wind_speeds=None):
        """
        Generate the wind speed frequencies from the Weibull parameters.  Use the
        cumulative form of the function and calculate the probability of the wind speed
        in a given bin via the difference in the cumulative function at the bin edges.
        Args:

            A (np.array): The Weibull A parameter.
            k (np.array): The Weibull k parameter.
            wind_speeds (np.array): The wind speeds to calculate the frequencies for.
                If None, the frequencies are calculated for 0 to 25 m/s in 1 m/s increments.
                Default is None.

        Returns:
            np.array: The wind speed frequencies.
        """

        if wind_speeds is None:
            wind_speeds = np.arange(0.0, 25.0, 1.0)

        # Define the wind speed edges
        ws_step = wind_speeds[1] - wind_speeds[0]
        wind_speed_edges = np.arange(
            wind_speeds[0] - ws_step / 2, wind_speeds[-1] + ws_step, ws_step
        )

        # Get the cumulative distribution function at the edges
        cdf_edges = self._weibull_cumulative(wind_speed_edges, A, k)

        # The frequency is the difference in the cumulative distribution function
        # at the edges
        freq = cdf_edges[1:] - cdf_edges[:-1]

        # Normalize the frequency
        # TODO: This is perhaps not quite right, if the user only asks
        # for wind speeds at say 8 and 10 m/s
        # then the bins would be from 7 to 11 and so fairly there is
        # distribution outside the range, but
        # I think this is on the whole right -- PF
        freq = freq / freq.sum()

        return wind_speeds, freq

    def get_wind_rose_at_point(self, x, y, wind_speeds=None, fixed_ti_value=0.06):
        """
        Get the wind rose at a given x, y location.  Interpolate the parameters to the point
        and then generate the wind rose.

        Args:
            x (float): The x location to interpolate.
            y (float): The y location to interpolate.
            wind_speeds (np.array): The wind speeds to calculate the frequencies for.
                If None, the frequencies are calculated for 0 to 25 m/s in 1 m/s increments.
                Default is None.
            fixed_ti_value (float): The fixed turbulence intensity value to use in the wind rose.
                Default is 0.06.
        """

        if wind_speeds is None:
            wind_speeds = np.arange(0.0, 25.0, 1.0)

        # Get the interpolated data
        sector_freq = self._interpolate_data(x, y, self.interpolant_sector_freq)
        weibull_A = self._interpolate_data(x, y, self.interpolant_weibull_A)
        weibull_k = self._interpolate_data(x, y, self.interpolant_weibull_k)

        # Initialize the freq_table
        freq_table = np.zeros((self.n_sectors, len(wind_speeds)))

        # First fill in the rows of the table using the weibull distributions,
        # weighted by the sector freq
        for sector in range(self.n_sectors):
            wind_speeds, freq = self._generate_wind_speed_frequencies_from_weibull(
                weibull_A[sector], weibull_k[sector], wind_speeds=wind_speeds
            )
            freq_table[sector, :] = sector_freq[sector] * freq

        # Normalize the table
        freq_table = freq_table / freq_table.sum()

        # Return the wind rose
        return WindRose(
            wind_directions=self.wind_directions,
            wind_speeds=wind_speeds,
            freq_table=freq_table,
            ti_table=fixed_ti_value,
        )

    def get_wind_rose_by_turbine(self, layout_x, layout_y, wind_speeds=None, fixed_ti_value=0.06):
        """
        Get the wind rose at each turbine location in the layout.

        Args:
            layout_x (np.array): The x locations of the turbines.
            layout_y (np.array): The y locations of the turbines.
            wind_speeds (np.array): The wind speeds to calculate the frequencies for.
                If None, the frequencies are calculated for 0 to 25 m/s in 1 m/s increments.
                Default is None.
            fixed_ti_value (float): The fixed turbulence intensity value to use in the wind rose.
                Default is 0.06.
        """

        if wind_speeds is None:
            wind_speeds = np.arange(0.0, 25.0, 1.0)

        # Initialize the list of wind roses
        wind_roses = []

        # Loop through the turbines and get the wind rose at each location
        for i in range(len(layout_x)):
            wind_rose = self.get_wind_rose_at_point(
                layout_x[i], layout_y[i], wind_speeds=wind_speeds, fixed_ti_value=fixed_ti_value
            )
            wind_roses.append(wind_rose)

        return WindRoseByTurbine(layout_x=layout_x, layout_y=layout_y, wind_roses=wind_roses)
