# page 4 before projection
import numpy as np
import matplotlib.pyplot as plt

from astropy.table import Table
from astropy.time import Time
import astropy.units as u

filename = "20250212-083213_TPI-GRAD300-SUN_01#_01#.fits" 
data = Table.read(filename)

jd = data["JD"]
time_sec = (jd - jd[0]) * 86400.0  # seconds since start

plt.figure(figsize=(8, 4))
plt.plot(time_sec, data["RIGHT_POL"])
plt.xlabel("Time since start (s)")
plt.ylabel("RIGHT_POL (arb. units)")
plt.title("Time Ordered Data (TOD): RIGHT_POL")
plt.xlim(0,4000)
plt.show()

#2D scatter map
x = data["Az_Offset"]    # deg
y = data["El_Offset"]    # deg
signal = data["RIGHT_POL"]

plt.figure(figsize=(6, 6))
sc = plt.scatter(
    x, y,
    c=signal,
    s=2,
)
plt.xlabel("Azimuth offset (deg)")
plt.ylabel("Elevation offset (deg)")
plt.title("TOD projected in horizontal offsets")
plt.colorbar(sc, label="RIGHT_POL (arb. units)")
plt.gca().invert_xaxis()  # common convention
plt.show()

#gal coord comparison
plt.figure(figsize=(6, 6))
sc = plt.scatter(
    data["Gal_Long"],
    data["Gal_Lat"],
    c=signal,
    s=2,
)
plt.xlabel("Galactic Longitude (deg)")
plt.ylabel("Galactic Latitude (deg)")
plt.title("TOD projected in Galactic coordinates")
plt.colorbar(sc, label="RIGHT_POL (arb. units)")
plt.show()

#Mapping TOD to a 2D Map
def project_tod_to_map(data, xcol, ycol, signal_col, npix=100):
    # Convert to plain numpy arrays
    x = np.array(data[xcol])
    y = np.array(data[ycol])
    signal = np.array(data[signal_col])
    
    # Remove NaNs or infinite values
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(signal)
    x = x[mask]
    y = y[mask]
    signal = signal[mask]
    
    # Define 2D grid edges
    x_edges = np.linspace(np.min(x), np.max(x), npix+1)
    y_edges = np.linspace(np.min(y), np.max(y), npix+1)
    
    # Compute hit map and signal sum
    hit_map, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])
    signal_sum, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges], weights=signal)
    
    # Mean signal per pixel
    with np.errstate(invalid='ignore', divide='ignore'):
        map_data = signal_sum / hit_map
    map_data[hit_map == 0] = np.nan  # empty pixels
    
    # Bin centers
    x_centers = 0.5*(x_edges[:-1] + x_edges[1:])
    y_centers = 0.5*(y_edges[:-1] + y_edges[1:])
    
    return map_data.T, hit_map.T, x_centers, y_centers

#sanity check
print(np.min(x), np.max(x))
print(np.min(y), np.max(y))
print(np.min(signal), np.max(signal))

#Example: project RIGHT_POL in horizontal coord.
map_data, hit_map, x_centers, y_centers = project_tod_to_map(
    data,
    xcol="Azimuth",
    ycol="Elevation",
    signal_col="RIGHT_POL",
    npix=100  #higher npix like 200 made it all white, lower npix like 25 closed the gaps or covered the empty pixels
)

plt.figure(figsize=(7,6))
plt.imshow(map_data, origin='lower',
           extent=[x_centers[0], x_centers[-1], y_centers[0], y_centers[-1]],
           aspect='auto', cmap='inferno')
plt.colorbar(label="RIGHT_POL (arb. units)")
plt.xlabel("Azimuth (deg)")
plt.ylabel("Elevation (deg)")
plt.title("Projected 2D Map (RIGHT_POL)")
plt.show()

plt.figure(figsize=(7,6))
plt.imshow(hit_map, origin='lower',
           extent=[x_centers[0], x_centers[-1], y_centers[0], y_centers[-1]],
           aspect='auto', cmap='viridis')
plt.colorbar(label="Number of samples per pixel")
plt.xlabel("Azimuth (deg)")
plt.ylabel("Elevation (deg)")
plt.title("Hit Map")
plt.show()

#Compare BBC channels:
map_bbc09, _, _, _ = project_tod_to_map(data, "Azimuth", "Elevation", "BBC09u", npix=100)
plt.imshow(map_bbc09, origin='lower', extent=[x_centers[0], x_centers[-1], y_centers[0], y_centers[-1]],
           aspect='auto', cmap='plasma')
plt.colorbar(label="BBC09u Power")
plt.title("BBC09u 2D Map")
plt.show()

map_bbc11, _, _, _ = project_tod_to_map(data, "Az_Offset", "El_Offset", "BBC11u", npix=100)
plt.imshow(map_bbc11, origin='lower', extent=[x_centers[0], x_centers[-1], y_centers[0], y_centers[-1]],
           aspect='auto', cmap='plasma')
plt.colorbar(label="BBC11u Power")
plt.title("BBC11u 2D Map")
plt.show()

map_bbc14, _, _, _ = project_tod_to_map(data, "Az_Offset", "El_Offset", "BBC14u", npix=100)
plt.imshow(map_bbc14, origin='lower', extent=[x_centers[0], x_centers[-1], y_centers[0], y_centers[-1]],
           aspect='auto', cmap='plasma')
plt.colorbar(label="BBC14u Power")
plt.title("BBC14u 2D Map")
plt.show()

#Mean BBC
bbc_cols = ["BBC09u", "BBC10u", "BBC11u", "BBC12u", "BBC13u", "BBC14u", "BBC15u", "BBC16u"]

# Compute mean signal per time sample
bbc_mean = np.mean([data[col] for col in bbc_cols], axis=0)

plt.figure(figsize=(8,4))
plt.plot(data["RIGHT_POL"], label="RIGHT_POL")
plt.plot(bbc_mean, "--", label="Mean of BBCs")
plt.xlabel("Time sample")
plt.ylabel("Power (arb. units)")
plt.legend()
plt.title("RIGHT_POL vs mean of BBC channels")
plt.show()

#Study effect of pixel/map size
for npix in [50, 100, 110, 120, 130, 140]:
    map_data, hit_map, x_centers, y_centers = project_tod_to_map(
        data, "Azimuth", "Elevation", "RIGHT_POL", npix=npix
    )
    plt.figure()
    plt.imshow(map_data, origin='lower', extent=[x_centers[0], x_centers[-1], y_centers[0], y_centers[-1]],
               aspect='auto', cmap='inferno')
    plt.colorbar(label="RIGHT_POL")
    plt.title(f"Pixel size test, npix={npix}")
    plt.show()
