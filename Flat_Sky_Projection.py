#page 4 : Flat Sky Projection application
from astropy.wcs import WCS
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy.time import Time
import astropy.units as u

filename = "20250212-083213_TPI-GRAD300-SUN_01#_01#.fits"
data = Table.read(filename)
print(data.colnames)

def tod_to_wcs_pixels(data, xcol='Az_Offset', ycol='El_Offset', npix=100, pixel_size=0.1, ref_point=(0,0)):
    """
    Converts sky coordinates in TPI data to pixel coordinates using a flat-sky TAN projection.
    
    Parameters
    ----------
    data : astropy.table.Table
        The TPI table containing coordinates and signal
    xcol : str
        Column name for the longitude coordinate (Azimuth or RA or GLON)
    ycol : str
        Column name for the latitude coordinate (Elevation or DEC or GLAT)
    npix : int
        Number of pixels along each axis (map will be square)
    pixel_size : float
        Pixel size in degrees
    ref_point : tuple of float
        Reference coordinate (olon, olat) in degrees
    
    Returns
    -------
    wcs : astropy.wcs.WCS
        The WCS object for the map
    x_pix : np.ndarray
        Pixel coordinates along X axis (0-indexed)
    y_pix : np.ndarray
        Pixel coordinates along Y axis (0-indexed)
    """
    # Reference point
    ref_olon, ref_olat = ref_point
    
    # Create WCS
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ['OLON-TAN', 'OLAT-TAN'] 
    wcs.wcs.crpix = [npix//2, npix//2]       
    wcs.wcs.crval = [ref_olon, ref_olat]     
    wcs.wcs.cdelt = [pixel_size, pixel_size]
    
    # Convert sky coordinates to pixel coordinates
    x = np.array(data[xcol])
    y = np.array(data[ycol])
    
    # Remove NaNs
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    
    x_pix, y_pix = wcs.all_world2pix(x, y, 0)  # 0-indexed
    
    return wcs, x_pix, y_pix, mask

#example
wcs, x_pix, y_pix, mask = tod_to_wcs_pixels(data, xcol='Az_Offset', ycol='El_Offset', npix=100, pixel_size=0.1)

# Use the mask to filter the signal
signal = np.array(data['RIGHT_POL'])[mask]

#project into 2d map
# Define empty map
map_data = np.full((100, 100), np.nan)
hit_map = np.zeros((100, 100))

# Loop over points (simple method)
for xi, yi, val in zip(x_pix, y_pix, signal):
    ix = int(np.floor(xi))
    iy = int(np.floor(yi))
    if 0 <= ix < 100 and 0 <= iy < 100:
        if np.isnan(map_data[iy, ix]):
            map_data[iy, ix] = val
        else:
            map_data[iy, ix] += val
        hit_map[iy, ix] += 1

# Compute mean in each pixel
with np.errstate(invalid='ignore', divide='ignore'):
    map_data = map_data / hit_map
map_data[hit_map == 0] = np.nan

plt.figure(figsize=(7,6))
plt.imshow(map_data, origin='lower', cmap='inferno', aspect='auto')
plt.colorbar(label='RIGHT_POL (arb. units)')
plt.xlabel('Pixel X')
plt.ylabel('Pixel Y')
plt.title('2D Map projected with Flat-Sky TAN WCS')
plt.show()

plt.figure(figsize=(7,6))
plt.imshow(hit_map, origin='lower', cmap='inferno', aspect='auto')
plt.colorbar(label='Number of samples')
plt.xlabel('Pixel X')
plt.ylabel('Pixel Y')
plt.title('Hit Map')
plt.show()
