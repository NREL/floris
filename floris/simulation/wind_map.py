# Copyright 2019 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy of the
# License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

from ..utilities import Vec3
import numpy as np
import scipy as sp


class WindMap():
    """
    WindMap contains the properies that characterize the initial state of the
    wind farm atmosphere.

    Args:
        -   **wind_speed**: A list that contains the wind speed  
                values (in m/s) at each measurement location.
        -   **wind_direction**: A list that contains the wind 
                direction values (in deg) at each measurement location.
        -   **turbulence_intensity**: A list that contains the 
                turbulence intensity values at each measurement 
                location (expressed as a decimal fraction).
        -   **layout_array**: An array that contains the
                x and y coordinates of each turbine.
        -   **wind_layout**: A tuple that contains the x and y 
                coordinates of the atmospheric measurement locations.
                
    Returns:
        WindMap: An instantiated WindMap object.
    """
    def __init__(self, wind_layout, layout_array, turbulence_intensity=None, wind_direction=None, wind_speed=None):
        self.wind_layout = wind_layout
        self.layout_array = layout_array
        self.input_direction = wind_direction
        self.input_speed = wind_speed
        self.input_ti = turbulence_intensity

        # Indicates that fix_wind_layout has been used when True. Initializing to False.
        self.duplicated_wind_layout = False

        if wind_direction:
            self.calculate_wind_direction()
        if wind_speed:
            self.calculate_wind_speed()
        if turbulence_intensity:
            self.calculate_turbulence_intensity()

    # Public functions
    def calculate_wind_speed(self, grid = None, interpolate = False):
        """
        This method calculates the wind speeds at each point, 
        interpolated and extrapolated from the input measurements.

        Args:
            -   **grid**: If set to True, this method calculates the
                resulting output values at gridpoint coordinates instead
                of turbine coordinates. Defaults to None.
            -   **interpolate**: A parameter that is switched to True
                if the wind measurement input values are not all
                equal. Defaults to False.

        Returns:
            *None* -- The values are updated directly in the
            :py:class:`floris.simulation.wind_map` object.
        """
        speed = self.input_speed
        if grid is not True: 
            if all(elem == speed[0] for elem in speed):
                self._turbine_wind_speed = [speed[0]]*len(self.layout_array[0])
            else:
                interpolate = True

        else:
            if all(elem == speed[0] for elem in speed):
                self._grid_wind_speed = np.full(np.shape(self.grid_layout[0]), speed[0])
            else:
                interpolate = True

        if interpolate is True:  
            if self.duplicated_wind_layout == True and len(self.input_speed) == len(self.wind_layout[0]) - len(self.input_speed):
                self._input_speed = self.input_speed + self.input_speed

            if grid is None:
                layout_array = np.array(self.layout_array)
                xp,yp = layout_array[0], layout_array[1]
                newpts= list(zip(xp, yp)) 
            else:
                layout_array = self.grid_layout
                xp,yp = layout_array[0], layout_array[1]
                newpts = layout_array
            x, y = self.wind_layout[0], self.wind_layout[1]
            z = np.array(speed)
            pts = list(zip(x, y))
            try: 
                interp = sp.interpolate.LinearNDInterpolator(pts, z, fill_value= np.nan)
            except Exception: 
                self.fix_wind_layout(recalculate = 'speed', grid = grid)
            else:
                zz = interp(newpts)
                idx = np.where(np.isnan(zz) == False)
                if np.shape(idx) == (xp.ndim, 0): 
                    near =  sp.interpolate.NearestNDInterpolator(pts, z)
                else:
                    nearpts = list(zip(xp[idx],yp[idx]))
                    near = sp.interpolate.NearestNDInterpolator(nearpts, zz[idx])
                wind_sp = interp(newpts)
                nearspeed = near(newpts)
                idx =np.where(np.isnan(wind_sp) == True)
                wind_sp[idx] = nearspeed[idx]
                if grid is None:  self._turbine_wind_speed = wind_sp.tolist()
                else: self._grid_wind_speed = wind_sp

    def calculate_wind_direction(self, grid = None, interpolate = False):
        """
        This method calculates the wind direction at each point, 
            interpolated and extrapolated from the input measurements.
            
        Args:
            -   **grid**: If set to True, this method calculates the 
                resulting output values at gridpoint coordinates instead 
                of turbine coordinates. Defaults to None.
            -   **interpolate**: A parameter that is switched to True
                if the wind measurement input values are not all 
                equal. Defaults to False.
        
        Returns:
            *None* -- The values are updated directly in the 
            :py:class:`floris.simulation.wind_map` object.
        """
        wdir = self.input_direction
        if grid is not True: 
            if all(elem == wdir[0] for elem in wdir):
                self._turbine_wind_direction = list((np.array([wdir[0] - 270]*len(self.layout_array[0])) % 360 + 360) % 360)
            else:
                interpolate = True
        else:
            if all(elem == wdir[0] for elem in wdir):
                self._grid_wind_direction = (np.full(np.shape(self.grid_layout[0]), wdir[0] - 270) % 360 + 360) % 360
            else:
                interpolate = True
                
        if interpolate is True: 
            if self.duplicated_wind_layout == True and len(self.input_direction) == len(self.wind_layout[0]) - len(self.input_direction):
                self._input_direction = self.input_direction + self.input_direction

            if grid is not True:
                layout_array = np.array(self.layout_array)
                xp,yp = layout_array[0], layout_array[1] 
                newpts= list(zip(xp, yp))
            else:
                xt, yt = np.array(self.layout_array)[0], np.array(self.layout_array)[1]
                new_turb_pts = list(zip(xt, yt))
                layout_array = self.grid_layout
                xp,yp = layout_array[0], layout_array[1] 
                newpts = layout_array
                turb_wd = np.zeros(np.shape(np.array(self.layout_array)))
            wind_layout_array = np.array(self.wind_layout)
            x, y = wind_layout_array[0], wind_layout_array[1]
            pts = list(zip(x, y))
            a = np.array(self.input_direction)
            wdxy =[np.sin(a * np.pi / 180), np.cos(a * np.pi / 180)]
            wd = np.zeros(np.shape(layout_array))
            for i in range(2):
                z = wdxy[i]              
                try: 
                    interp = sp.interpolate.LinearNDInterpolator(pts, z, fill_value= np.nan)
                except Exception: 
                    self.fix_wind_layout(recalculate = 'direction', grid = grid)
                else:
                    zz = interp(newpts)
                    idx =np.where(np.isnan(zz) == False)
                    if np.shape(idx) == (xp.ndim, 0): 
                        near =  sp.interpolate.NearestNDInterpolator(pts, z)
                    else:
                        nearpts = list(zip(xp[idx],yp[idx]))
                        near = sp.interpolate.NearestNDInterpolator(nearpts, zz[idx])
                    if grid == True: turb_wd[i] = near(new_turb_pts)
                    wind_dir = interp(newpts)
                    neardir = near(newpts)
                    idx = np.where(np.isnan(wind_dir) == True)
                    wind_dir[idx] = neardir[idx]
                    wd[i] = wind_dir 
            widi = (np.arctan2(wd[0] , wd[1]) * 180 / np.pi)  - 270 % 360
            widi =  (widi + 360) % 360
            if grid is not True: 
                self._turbine_wind_direction = widi.tolist()
            else:
                twidi = (np.arctan2(turb_wd[0] ,turb_wd[1]) * 180 / np.pi) - 270 % 360
                twidi = (twidi + 360) % 360

                self._turbine_wind_direction =twidi.tolist()
                self._grid_wind_direction = widi

    def calculate_turbulence_intensity(self, grid = None, interpolate = False):
        """
        This method calculates the turbulence intensity at each turbine, 
            interpolated and extrapolated from the input measurements.
            
        Args:
            -   **grid**: If set to True, this method calculates the 
                resulting output values at gridpoint coordinates instead 
                of turbine coordinates. Defaults to None.
            -   **interpolate**: A parameter that is switched to True
                if the wind measurement input values are not all 
                equal. Defaults to False.
        
        Returns:
            *None* -- The values are updated directly in the 
            :py:class:`floris.simulation.wind_map` object.
        """
        ti = self.input_ti
        if grid is not True: 
            if all(elem == ti[0] for elem in ti):
                self._turbine_turbulence_intensity = [ti[0]]*len(self.layout_array[0])
            else:
                interpolate = True
        else:
            ti = self.input_ti
            if all(elem == ti[0] for elem in ti):
                self._grid_turbulence_intensity = np.full(np.shape(self.grid_layout[0]), ti[0])
            else:
                interpolate = True
                
        if interpolate is True:
            if self.duplicated_wind_layout == True and len(self.input_ti) == len(self.wind_layout[0]) - len(self.input_ti):
                self._input_ti = self.input_ti + self.input_ti

            if grid is None:
                layout_array = np.array(self.layout_array)
                xp,yp = layout_array[0], layout_array[1]
                newpts= list(zip(xp, yp))
            else:
                layout_array = self.grid_layout
                xp,yp = layout_array[0], layout_array[1]
                newpts = layout_array
            wind_layout_array = np.array(self.wind_layout)   
            x,y = wind_layout_array[0], wind_layout_array[1]
            z = np.array(ti)
            pts = list(zip(x, y))
            try: 
                interp = sp.interpolate.LinearNDInterpolator(pts, z, fill_value = np.nan)
            except Exception: 
                self.fix_wind_layout(recalculate = 'direction', grid = grid)
            else:
                zz = interp(newpts)
                idx =np.where(np.isnan(zz) == False)
                if np.shape(idx) == (xp.ndim, 0): 
                    near =  sp.interpolate.NearestNDInterpolator(pts, z)
                else:
                    nearpts = list(zip(xp[idx],yp[idx]))
                    near = sp.interpolate.NearestNDInterpolator(nearpts, zz[idx])
                wind_t = interp(newpts)
                neart = near(newpts)
                idx = np.where(np.isnan(wind_t) == True)
                wind_t[idx] = neart[idx]
                if grid is None: 
                    self._turbine_turbulence_intensity = wind_t.tolist() 
                else:  
                    self._grid_turbulence_intensity = wind_t

    def fix_wind_layout(self, recalculate, grid=None):
        """
        This method analyzes interpolation errors having to do with 
        the wind layout coordinates, and provides a solution.
            
        Args:
            -   **recalculate**: A string that indicates which 
                method was interupted in order to fix the
                wind layout. The indicated method will be restarted 
                after the inputs have been fixed, if not a case of 
                unequal input lengths.
            -   **grid**: Boolean. When True, this indicates that
                the method being interrupted is interpolating 
                is calculating values at flow field grid points. 
                Defaults to None.
        
        Returns:
            *None* -- The values are updated directly in the 
            :py:class:`floris.simulation.wind_map` object.
        """
        x,y = self.wind_layout[0], self.wind_layout[1]
        xp,yp = x,y
        array_length = len(x)
        
        # check for case of unequal input lengths, stop all calculations if true
        if array_length != len(y):
            raise SystemExit('FLORIS has stopped because the number of wind input coordinates is not\
                \nequal in the x and y directions.')

        elif len(self.input_speed) > 1 and len(self.input_speed) != array_length : 
            raise SystemExit('FLORIS has stopped because the number of wind input coordinates is not\
            \nequal to the number of wind input measurements.')

        elif len(self.input_direction) > 1 and len(self.input_direction) != array_length : 
            raise SystemExit('FLORIS has stopped because the number of wind input coordinates is not\
            \nequal to the number of wind input measurements.')
        
        elif len(self.input_ti) > 1 and len(self.input_ti) != array_length : 
            raise SystemExit('FLORIS has stopped because the number of wind input coordinates is not\
            \nequal to the number of wind input measurements.')

        # assume error is either:
        # QhullError: QH6214 qhull input error: not enough points(2) to construct initial simplex (need 4)
        # or QhullError: QH6154 Qhull precision error: Initial simplex is flat (facet 1 is coplanar with the interior point)
        # both errors are avoided by adding line of inputs parrell to line originally given
        else:
            # find slope of original wind_layout coords
            j, dx, dy = 0, 0, 0
            while j<= array_length and dx == 0 and dy == 0:
                dx, dy = (x[j+1]-x[j]), (y[j+1] - y[j])
                j = j+1

            # add identical inputs in parallel new line by offsetting points perpendicular to original line
            if dx == 0:
                xp = [min(self.layout_array[0]) - 10]*array_length + [max(self.layout_array[0]) + 10]*array_length
                yp = y + y

            elif dy == 0: 
                xp = x + x
                yp = [min(self.layout_array[1]) - 10]*array_length + [max(self.layout_array[1]) + 10]*array_length
   
            else:
                dist = np.sqrt((dx)**2 + (dy)**2)
                scaling_param = 500/dist
                # TODO: find best offset distance (determined by # of turbines or size of domain?)
                # problems with LinearNDInterpolator when offset is small and grid res is high (still runs NearestNDInterpolator)
                for i in range(array_length):
                    xp = xp +[x[i] - (dy* scaling_param)]
                    yp = yp +[y[i] + (dx* scaling_param)]

            # reset input parameters
            if self.duplicated_wind_layout == False: 
                self.wind_layout = (xp,yp)
                if len(self.input_direction) != 1: 
                    self.input_direction = self.input_direction + self.input_direction
                if len(self.input_ti) != 1: 
                    self.input_ti = self.input_ti + self.input_ti
                if len(self.input_speed) != 1: 
                    self.input_speed = self.input_speed + self.input_speed

            self.duplicated_wind_layout = True
            
            # restart calculation with new inputs
            if recalculate == 'turbulence': 
                self.calculate_turbulence_intensity(grid=grid)
            elif recalculate == 'direction': 
                self.calculate_wind_direction(grid=grid)
            elif recalculate == 'speed': 
                self.calculate_wind_speed(grid=grid)
            else: pass

    # Getters & Setters

    @property       
    def turbine_wind_speed(self):
        """
        This property returns the wind speeds at each turbine.
        
        Returns:
            A list of wind speeds at each turbine in m/s.
        """   
        return self._turbine_wind_speed
    
    @property       
    def grid_wind_speed(self):
        """
        This property returns the wind speeds at each gridpoint.
        
        Returns:
            An array of wind speeds at each turbine in m/s.
        """   
        return self._grid_wind_speed

    @property
    def turbine_wind_direction(self):
        """
        This property returns the wind direction at each turbine. 
        
        Returns:
            A list of wind directions at each turbine in degrees.
        """   
        return self._turbine_wind_direction
    
    @property
    def grid_wind_direction(self):
        """
        This property returns the wind direction at each gridpoint.

        Returns:
            An array of wind directions at each gridpoint in degrees.
        """   
        return self._grid_wind_direction
    
    @property       
    def turbine_turbulence_intensity(self):
        """
        This property returns the turbulence intensity at each turbine

        Returns:
            A list of turbulence intensities at each turbine in decimal fractions.
        """
        return self._turbine_turbulence_intensity

    @property       
    def grid_turbulence_intensity(self):
        """
        This property returns the turbulence intensity at each gridpoint.
        
        
        Returns:
            An array of the turbulence intensity at each gridpoint as a decimal fraction.
        """
        return self._grid_turbulence_intensity

    @property   
    def input_direction(self):
        """
        This property stores and returns the wind directions at each 
            input measurement location.

        Returns:
            A list of wind directions expressed in degrees.
        """
        return self._input_direction

    @input_direction.setter
    def input_direction(self, value):
        self._input_direction = value
    
    @property
    def input_speed(self):
        """
        This property stores and returns the wind speeds at each 
            input measurement location.

        Returns:
            A list of wind speeds expressed in meters per second.
        """
        return self._input_speed

    @input_speed.setter
    def input_speed(self, value):
        self._input_speed = value
   
    @property
    def input_ti(self):
        """
        This property stores and returns the turbulence intensity 
            at each input measurement location.

        Returns:
            A list of turbulence intensities expressed in decimal fraction.
        """
        return self._input_ti

    @input_ti.setter
    def input_ti(self, value):
        self._input_ti = value

    @property
    def grid_layout(self):
        """
        This property stores and returns an array that contains the
            x and y coordinates of each gridpoint in the flow field.

        Returns:
            An array containing the x and y coordinates for each 
            gridpoint location.
        """
        return self._grid_layout

    @grid_layout.setter
    def grid_layout(self, value):
        self._grid_layout = value

    @property
    def wind_layout(self):
        """
        This property stores and returns a tuple that contains the
            x and y coordinates of each wind measurement location.

        Returns:
            A tuple of coordinates for each wind measurement location.
        """
        return self._wind_layout

    @wind_layout.setter
    def wind_layout(self, value):
        self._wind_layout = value
