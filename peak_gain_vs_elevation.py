# Page 5 : Study the Gain Pattern w.r.t. Elevations 
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table

files = [
    r"20260204-100711_TPI-PROJ01-SUN_10#_01#.fits",
    r"20260204-110759_TPI-PROJ01-SUN_11#_01#.fits",
    r"20260204-121429_TPI-PROJ01-SUN_12#_01#.fits"
]

peak_elevations = []
peak_gains = []
labels = []

for file in files:
    with fits.open(file) as hdul:
        data = Table(hdul[1].data)

    elevation = np.array(data["Elevation"])
    power = np.array(data["RIGHT_POL"])

    # Remove NaNs
    mask = np.isfinite(elevation) & np.isfinite(power)
    elevation = elevation[mask]
    power = power[mask]

    # Find peak
    peak_index = np.argmax(power)
    elev_at_peak = elevation[peak_index]
    peak_gain = power[peak_index]

    peak_elevations.append(elev_at_peak)
    peak_gains.append(peak_gain)

    # Extract hour from filename (10#, 11#, 12# → 10, 11, 12)
    hour_utc = int(file.split("-SUN_")[1].split("#")[0])
    hour_cet = hour_utc + 1   # UTC → CET (February = UTC+1)

    labels.append(f"Sun at {hour_cet}h (CET)")

# Convert to arrays
peak_elevations = np.array(peak_elevations)
peak_gains = np.array(peak_gains)

# Plot
plt.figure(figsize=(8,5))

colors = ['blue', 'green', 'red']

for i in range(len(files)):
    plt.scatter(
        peak_elevations[i],
        peak_gains[i],
        color=colors[i],
        s=100,
        label=labels[i]
    )

plt.plot(peak_elevations, peak_gains, linestyle='--', alpha=0.5)

plt.xlabel("Elevation at Peak (deg)")
plt.ylabel("Peak Gain (arb. units)")
plt.title("Sun Peak Gain vs Elevation")
plt.legend()
plt.grid(True)
plt.show()
