# PAGE 4 : TOD Curation
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy.stats import sigma_clip

filename = "20250212-083213_TPI-GRAD300-SUN_01#_01#.fits"
data = Table.read(filename)

# Time in seconds
time_sec = (data['JD'] - data['JD'][0]) * 86400.0

# Telescope speed & sigma clipping
dAz = np.gradient(np.array(data['Azimuth']))
dEl = np.gradient(np.array(data['Elevation']))
dt = np.gradient(time_sec)

speed = np.sqrt(dAz**2 + dEl**2) / dt

# Sigma clipping
speed_masked = sigma_clip(speed, sigma=3, maxiters=5)
good_points = ~speed_masked.mask

clean_data = data[good_points]
time_clean = time_sec[good_points]

plt.figure()
plt.plot(time_sec, speed, label="Original")
plt.plot(time_clean, speed[good_points], '.', label="Sigma-clipped")
plt.xlabel("Time (s)")
plt.ylabel("Telescope speed (deg/s)")
plt.legend()
plt.title("Sigma Clipping on Telescope Speed")
plt.show()

# BBC inspection
bbc_cols = ["BBC09u","BBC10u","BBC11u","BBC12u",
            "BBC13u","BBC14u","BBC15u","BBC16u"]

plt.figure(figsize=(10,4))
for col in bbc_cols:
    plt.plot(time_clean, clean_data[col], alpha=0.7, label=col)

plt.xlabel("Time (s)")
plt.ylabel("Power (arb. units)")
plt.title("BBC Channels Inspection")
plt.legend(ncol=2)
plt.show()

# Normalize using (Signal - Continuum) / (Peak - Continuum)
bbc_data = np.vstack([clean_data[col] for col in bbc_cols]).T
right_pol = clean_data["RIGHT_POL"]

threshold = np.percentile(right_pol, 90)
off_sun = right_pol <= threshold

bbc_data_norm = np.zeros_like(bbc_data)

plt.figure(figsize=(10,4))

for i, col in enumerate(bbc_cols):
    signal = bbc_data[:, i]

    # Continuum estimate (robust)
    continuum = np.median(signal[off_sun])

    # Peak estimate (robust)
    peak = np.percentile(signal, 99)

    delta = peak - continuum

    if delta == 0:
        norm_signal = np.zeros_like(signal)
    else:
        norm_signal = (signal - continuum) / delta

    bbc_data_norm[:, i] = norm_signal

    plt.plot(time_clean, norm_signal, alpha=0.7, label=col)

plt.xlabel("Time (s)")
plt.ylabel("Normalized Power") #((Signal - Continuum) / (Peak - Continuum))
plt.title("BBC Channels Properly Normalized")
plt.legend(ncol=2)
plt.show()

# Cross-correlation matrix
corr_matrix = np.corrcoef(bbc_data_norm, rowvar=False)

plt.figure(figsize=(6,5))
im = plt.imshow(corr_matrix, vmin=0, vmax=1)
plt.colorbar(im, label="Correlation coefficient")
plt.xticks(range(len(bbc_cols)), bbc_cols, rotation=45)
plt.yticks(range(len(bbc_cols)), bbc_cols)
plt.title("BBC Cross-Correlation Matrix")
plt.tight_layout()
plt.show()

# Reconstruct RIGHT_POL from BBC mean
bbc_mean = np.mean(bbc_data, axis=1)

plt.figure(figsize=(8,4))
plt.plot(time_clean, clean_data['RIGHT_POL'], label='RIGHT_POL')
plt.plot(time_clean, bbc_mean, '--', label='Mean of BBCs')
plt.xlabel("Time (s)")
plt.ylabel("Power (arb. units)")
plt.title("RIGHT_POL vs Mean BBC")
plt.legend()
plt.show()
