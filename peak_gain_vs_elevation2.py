import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
from scipy.optimize import curve_fit

files = [
    "20260204-100711_TPI-PROJ01-SUN_10#_01#.fits",
    "20260204-110759_TPI-PROJ01-SUN_11#_01#.fits",
    "20260204-121429_TPI-PROJ01-SUN_12#_01#.fits"
]

bbc_cols = ["BBC09u","BBC10u","BBC11u","BBC12u",
            "BBC13u","BBC14u","BBC15u","BBC16u"]

peak_elevations = []
peak_gains = []
labels = []

bbc_peak_matrix = []   # store BBC peak gains per file

# Extract peak gain per file
for file in files:
    with fits.open(file) as hdul:
        data = Table(hdul[1].data)

    elevation = np.array(data["Elevation"])
    power = np.array(data["RIGHT_POL"])

    mask = np.isfinite(elevation) & np.isfinite(power)
    elevation = elevation[mask]
    power = power[mask]

    peak_index = np.argmax(power)
    elev_at_peak = elevation[peak_index]
    peak_gain = power[peak_index]

    peak_elevations.append(elev_at_peak)
    peak_gains.append(peak_gain)

    # BBC peak extraction
    bbc_peaks = []
    for col in bbc_cols:
        bbc_signal = np.array(data[col])[mask]
        bbc_peaks.append(np.max(bbc_signal))
    bbc_peak_matrix.append(bbc_peaks)

    hour = file.split("-SUN_")[1].split("#")[0]
    labels.append(f"Sun at {hour}h")

peak_elevations = np.array(peak_elevations)
peak_gains = np.array(peak_gains)
bbc_peak_matrix = np.array(bbc_peak_matrix)

# POLYNOMIAL FIT (systematic trend with elevation)
poly_coeff = np.polyfit(peak_elevations, peak_gains, 2)
poly_model = np.poly1d(poly_coeff)

elev_fit = np.linspace(min(peak_elevations),
                       max(peak_elevations), 200)

plt.figure(figsize=(8,5))

for i in range(len(files)):
    plt.scatter(peak_elevations[i],
                peak_gains[i],
                s=100,
                label=labels[i])

plt.plot(elev_fit,
         poly_model(elev_fit),
         '--',
         label="Polynomial Fit (2nd order)")

plt.xlabel("Elevation (deg)")
plt.ylabel("Peak Gain (arb. units)")
plt.title("Gain vs Elevation – Polynomial Fit")
plt.legend()
plt.grid(True)
plt.show()

# FREQUENCY DEPENDENCE (BBC comparison)
plt.figure(figsize=(8,5))

for i, col in enumerate(bbc_cols):
    plt.plot(peak_elevations,
             bbc_peak_matrix[:, i],
             marker='o',
             label=col)

plt.xlabel("Elevation (deg)")
plt.ylabel("Peak Gain (arb. units)")
plt.title("BBC Peak Gain vs Elevation (Frequency Dependence)")
plt.legend(ncol=2)
plt.grid(True)
plt.show()
