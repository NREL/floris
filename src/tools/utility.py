"""utility module
"""
import numpy as np

def wrap_180(x):
    """
    Wrap an angle to between -180 and 180
    """  
    
    x = np.where(x<=-180.,x+360.,x)
    x = np.where(x>180.,x-360.,x)

    return(x)

def wrap_360(x):
    """
    Wrap an angle to between 0 and 360
    """  
    
    x = np.where(x<0.,x+360.,x)
    x = np.where(x>=360.,x-360.,x)

    return(x)

