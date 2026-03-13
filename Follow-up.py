import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy.stats import sigma_clip
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter

filename = "20250212-083213_TPI-GRAD300-SUN_01#_01#.fits"
data = Table.read(filename)

# Time in seconds
time_sec = (data['JD'] - data['JD'][0]) * 86400.0

# Telescope coordinate differences
dAz = np.gradient(np.array(data['Azimuth']))
dEl = np.gradient(np.array(data['Elevation']))
dt = np.gradient(time_sec)

# Telescope speed in deg/s
speed = np.sqrt(dAz**2 + dEl**2) / dt

# Sigma clipping to remove bad points
speed_masked = sigma_clip(speed, sigma=3, maxiters=5)
good_points = ~speed_masked.mask
clean_data = data[good_points]

# BBC data processing
bbc_cols = ["BBC09u","BBC10u","BBC11u","BBC12u",
            "BBC13u","BBC14u","BBC15u","BBC16u"]

bbc_data = np.vstack([clean_data[col] for col in bbc_cols]).T
mask = np.all(np.isfinite(bbc_data), axis=1)
bbc_data = bbc_data[mask]

# Mean BBC signal (total power)
bbc_mean = np.mean(bbc_data, axis=1)

# Savitzky–Golay filtering
window_length = 31
poly_order = 3
bbc_sg = savgol_filter(bbc_mean, window_length, poly_order)

# Beam map construction
az = np.array(clean_data['Az_Offset'])[mask]
el = np.array(clean_data['El_Offset'])[mask]
power = bbc_mean  # use BBC mean as power

npix = 100
az_edges = np.linspace(np.percentile(az, 1), np.percentile(az, 99), npix)
el_edges = np.linspace(np.percentile(el, 1), np.percentile(el, 99), npix)

hits, _, _ = np.histogram2d(az, el, bins=[az_edges, el_edges])
power_sum, _, _ = np.histogram2d(az, el, bins=[az_edges, el_edges], weights=power)
beam_map = np.divide(power_sum, hits, where=hits>0)
beam_map[hits == 0] = np.nan

# Gaussian 2D model
def gaussian_2d(coords, A, x0, y0, sx, sy, C):
    x, y = coords
    return (A * np.exp(-(((x-x0)**2)/(2*sx**2) + ((y-y0)**2)/(2*sy**2))) + C).ravel()

# Grid for fitting
X, Y = np.meshgrid((az_edges[:-1] + az_edges[1:]) / 2,
                   (el_edges[:-1] + el_edges[1:]) / 2)

mask_fit = ~np.isnan(beam_map)
xdata = X[mask_fit]
ydata = Y[mask_fit]
zdata = beam_map[mask_fit]

# Initial guess
p0 = [np.nanmax(beam_map), 0, 0, 0.5, 0.5, np.nanmedian(beam_map)]

popt_before, _ = curve_fit(gaussian_2d, (xdata, ydata), zdata, p0=p0)
A_b, x0_b, y0_b, sx_b, sy_b, C_b = popt_before

# FWHM
FWHM_x_b = 2*np.sqrt(2*np.log(2))*sx_b
FWHM_y_b = 2*np.sqrt(2*np.log(2))*sy_b

# Apply filtering to beam map (power replaced by filtered power)
power_filt = bbc_sg
power_sum_filt, _, _ = np.histogram2d(az, el, bins=[az_edges, el_edges], weights=power_filt)
beam_map_filt = np.divide(power_sum_filt, hits, where=hits>0)
beam_map_filt[hits == 0] = np.nan

mask_fit2 = ~np.isnan(beam_map_filt)
xdata2 = X[mask_fit2]
ydata2 = Y[mask_fit2]
zdata2 = beam_map_filt[mask_fit2]

popt_after, _ = curve_fit(gaussian_2d, (xdata2, ydata2), zdata2, p0=p0)
A_a, x0_a, y0_a, sx_a, sy_a, C_a = popt_after

FWHM_x_a = 2*np.sqrt(2*np.log(2))*sx_a
FWHM_y_a = 2*np.sqrt(2*np.log(2))*sy_a

# SNR estimation
noise_before = np.std(bbc_mean - bbc_sg)
noise_after = np.std(bbc_sg - savgol_filter(bbc_sg, 31, 3))

SNR_before = np.percentile(bbc_mean, 99) / noise_before
SNR_after = np.percentile(bbc_sg, 99) / noise_after

# Print results
print("\n=== Peak amplitude ===")
print("Before filtering:", np.percentile(bbc_mean, 99))
print("After filtering :", np.percentile(bbc_sg, 99))

print("\n=== FWHM comparison ===")
print("FWHM_x before:", FWHM_x_b, "deg")
print("FWHM_y before:", FWHM_y_b, "deg")
print("FWHM_x after :", FWHM_x_a, "deg")
print("FWHM_y after :", FWHM_y_a, "deg")

print("\n=== Gaussian fit peak ===")
print("Peak before:", A_b)
print("Peak after :", A_a)

print("\n=== SNR improvement ===")
print("SNR before:", SNR_before)
print("SNR after :", SNR_after)
print("Improvement factor:", SNR_after / SNR_before)

plt.figure(figsize=(10,5))
plt.plot(bbc_mean, label="Original BBC mean", alpha=0.6)
plt.plot(bbc_sg, label="Filtered BBC mean", linewidth=2)
plt.legend(); plt.grid(); plt.title("BBC Mean Filtering")
plt.show()

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.title("Beam map (before filtering)")
plt.imshow(beam_map.T, origin='lower')
plt.colorbar()

plt.subplot(1,2,2)
plt.title("Beam map (after filtering)")
plt.imshow(beam_map_filt.T, origin='lower')
plt.colorbar()
plt.show()