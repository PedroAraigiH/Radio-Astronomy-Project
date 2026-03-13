#Page 5
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy.stats import sigma_clip
from scipy.optimize import curve_fit

filename = "20260204-121429_TPI-PROJ01-SUN_12#_01#.fits"
data = Table.read(filename)
print(data.colnames)

# Time in seconds
time_sec = (data['JD'] - data['JD'][0]) * 86400.0

# Telescope coordinate differences
dAz = np.gradient(np.array(data['Azimuth']))
dEl = np.gradient(np.array(data['Elevation']))
dt = np.gradient(time_sec)

# Telescope speed in deg/s
speed = np.sqrt(dAz**2 + dEl**2) / dt

# Sigma clipping to identify outliers in speed:
# Sigma clip at 3 sigma
speed_masked = sigma_clip(speed, sigma=3, maxiters=5)

# Flag good points
good_points = ~speed_masked.mask
clean_data = data[good_points]

az = np.array(clean_data['Az_Offset'])
el = np.array(clean_data['El_Offset'])
#power = np.array(clean_data['RIGHT_POL'])
power = np.array(clean_data['BBC10u'])

# Define grid
npix = 100
az_edges = np.linspace(np.percentile(az, 1),
                       np.percentile(az, 99), npix)

el_edges = np.linspace(np.percentile(el, 1),
                       np.percentile(el, 99), npix)

# Hit map
hits, _, _ = np.histogram2d(az, el, bins=[az_edges, el_edges])

# Weighted sum
power_sum, _, _ = np.histogram2d(az, el, bins=[az_edges, el_edges], weights=power)

# Mean map
beam_map = np.divide(power_sum, hits, where=hits>0)
beam_map[hits == 0] = np.nan

plt.figure(figsize=(6,5))
plt.imshow(
    beam_map.T,
    origin='lower',
    extent=[az_edges[0], az_edges[-1], el_edges[0], el_edges[-1]]
)
plt.xlabel("Azimuth Offset (deg)")
plt.ylabel("Elevation Offset (deg)")
plt.title("Measured Beam Pattern (Sun scan)")
plt.colorbar(label="Power")
plt.show()

# Construct the Beam Profile :
def gaussian_2d(coords, A, x0, y0, sigma_x, sigma_y, C):
    x, y = coords
    return (
        A * np.exp(
            -(((x - x0)**2) / (2*sigma_x**2)
              + ((y - y0)**2) / (2*sigma_y**2))
        )
        + C
    ).ravel()


X, Y = np.meshgrid(
    (az_edges[:-1] + az_edges[1:]) / 2,
    (el_edges[:-1] + el_edges[1:]) / 2
)

# Remove NaNs
mask = ~np.isnan(beam_map)

xdata = X[mask]
ydata = Y[mask]
zdata = beam_map[mask]

# Initial Guess
A0 = np.nanmax(beam_map)
x0_0 = 0
y0_0 = 0
sigma_x0 = 0.5
sigma_y0 = 0.5
C0 = np.nanmedian(beam_map)

p0 = [A0, x0_0, y0_0, sigma_x0, sigma_y0, C0]

popt, pcov = curve_fit(
    gaussian_2d,
    (xdata, ydata),
    zdata,
    p0=p0
)

A, x0, y0, sigma_x, sigma_y, C = popt

# Convert sigma to FWHM
FWHM_x = 2*np.sqrt(2*np.log(2)) * sigma_x
FWHM_y = 2*np.sqrt(2*np.log(2)) * sigma_y

print("sigma x y:", sigma_x, sigma_y)
print("Peak Gain:", A)
print("Beam center:", x0, y0)
print("FWHM Az:", FWHM_x, "deg")
print("FWHM El:", FWHM_y, "deg")

# Asymmetry
ellipticity = FWHM_y / FWHM_x
print("Ellipticity ratio:", ellipticity)

# Visualize
fitted = gaussian_2d((X, Y), *popt).reshape(X.shape)

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.title("Measured Beam")
plt.imshow(beam_map.T, origin='lower')
plt.colorbar()

plt.subplot(1,2,2)
plt.title("Fitted Gaussian")
plt.imshow(fitted.T, origin='lower')
plt.colorbar()

plt.tight_layout()
plt.show()

# Known antenna gain in dBi
gain_dBi = 22

# Convert gain to linear scale
gain_linear = 10**(gain_dBi / 10)

# Example: measured power array (replace with your data)
P_measured = np.array(clean_data["RIGHT_POL"])

# Compute real power
P_real = P_measured / gain_linear

print("Linear gain factor:", gain_linear)
print("First 5 measured powers:", P_measured[:5])
print("First 5 real powers:", P_real[:5])

peak_measured = np.max(P_measured)
peak_real = peak_measured / gain_linear

print("Measured peak power:", peak_measured)
print("Real peak power:", peak_real)