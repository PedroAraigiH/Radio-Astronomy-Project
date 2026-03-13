#Ploting the spectrum.
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

# File name
filename = "20250212-083051_SPECTRUM-GRAD300-SUN_01#_01#.fits"

# Open FITS file
with fits.open(filename) as hdul:
    data = hdul[1].data
    header = hdul[1].header

# Print column names to check structure
print(data.columns)

power = data['RIGHT_POL']
freq = (header['BASEFREQ']+np.arange(power.shape[1])* header['BNDRES'])/1e6      

# Plot spectrum
plt.figure(figsize=(8,5))
plt.plot(freq, power[np.argmax(np.mean(power, axis=1))])
plt.xlabel("Frequency (MHz)")
plt.ylabel("Intensity (ADU)")
plt.title("Sun Spectrum")
plt.grid(True)
plt.legend(["Spectrum"])
plt.tight_layout()
plt.show()