import numpy as np

def vPositionZFTK(strip_index, pitch=1., N_sipm_arrays=3, Nstrips1=32, sgn=1):
    """
    Function to compute position of strips considering the real array geometry: i.e. gaps
    
    Parameters:
    -----------
    -strip_index: (int) strip position in its own plane
    -pitch: (float) readout pitch in mm
    -N_sipm_arrays: (int) nr of Hamamatsu arrays on the pcb (e.g. zirettino 3, petrioc-ftk 1)
    -Nstrips1: (int) number of strips of the single array (e.g. OR4 32, OR1 128)
    -sgn: (int) according to position frame of reference axis wrt strip number
        +1 same direction
        -1 oppostie direction
        
    Returns:
    --------
    -geometrical position in mm
    """
    
    
    assert -0.1 <= strip_index <= (N_sipm_arrays*Nstrips1-1+.1)

    #Define geometrical parameters of the SiPM array
    int_gap = 0.220 #mm
    ext_gap = 0.380 #mm
    
    #Distance btw the origin and the center of the nearest SiPM array (if more than 1 exists)
    d = int_gap + ext_gap + pitch * Nstrips1 
    # print(f"Displacement unit (d) {d} .")
    
    #Index of the SiPM array where the strip with index strip_index belongs  
    sipm_array_index = int(strip_index / Nstrips1)  
    # print(f"sipm_array_index {sipm_array_index} .")
    
    sipm_array_center_position = sgn * d * 0.5 * (-N_sipm_arrays + 1 + 2 * sipm_array_index)
    # print(f"sipm_array_center_position {sipm_array_center_position} .")
    
    internal_strip_index = strip_index - sipm_array_index * Nstrips1
    # print(f"internal_strip_index {internal_strip_index} .")
    
    if internal_strip_index >= Nstrips1 / 2:
        internal_strip_position = sgn * (-pitch * 0.5 * (Nstrips1 - 1 - 2 * internal_strip_index) + 0.5 * int_gap)
    else:
        internal_strip_position = sgn * (-pitch * 0.5 * (Nstrips1 - 1 - 2 * internal_strip_index) - 0.5 * int_gap) 
    
    return sipm_array_center_position + internal_strip_position #mm

PositionZFTK = np.vectorize(vPositionZFTK, otypes=[np.float], cache=False)