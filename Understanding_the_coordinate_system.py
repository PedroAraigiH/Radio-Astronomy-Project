#pages 2 and 3
from astropy.io import fits
import matplotlib.pyplot as plt
from astropy.table import Table

filename = "20250212-083216_IMAGE-GRAD300-SUN_01#_01#.fits"
header = fits.getheader(filename) # for an IMAGE file

data = fits.getdata(filename) # for an IMAGE file
#print(data)

plt.imshow(data)
plt.show()

filename2 = "20250212-083213_TPI-GRAD300-SUN_01#_01#.fits" 
data = Table.read(filename2)
print(data.colnames)

plt.plot(data["Az_Offset"], data["El_Offset"])
plt.show()
plt.plot(data["Azimuth"], data["Elevation"])
plt.show()
plt.plot(data["Azimuth"], data["Elevation"])
plt.show()

from astropy.coordinates import EarthLocation
import astropy.units as u

grad300_location = EarthLocation(
    lat=43.93300 * u.deg,
    lon=5.71530 * u.deg,
    height=654.8 * u.m
)

from astropy.time import Time

jd = data["JD"]
t = Time(jd, format="jd", scale="utc")

print("Start time (UTC):", t[0].isot)
print("End time (UTC):", t[-1].isot)

print(t[0].iso)
print(t[-1].iso)
print("Duration (s):", (t[-1] - t[0]).to(u.s))

header2 = fits.getheader(filename2)

############## Page 3 ##################

from astropy.coordinates import AltAz, SkyCoord
import astropy.units as u

print(data["Azimuth"].unit) #check units
print(data["Elevation"].unit)

altaz_frame = AltAz(
    obstime=t,
    location=grad300_location
)

coords_altaz = SkyCoord(
    az=data["Azimuth"],
    alt=data["Elevation"],
    frame=altaz_frame
)

coords_gal = coords_altaz.galactic

plt.figure()
plt.plot(coords_gal.l.deg, coords_gal.b.deg, ".", label="Computed")
plt.plot(data["Gal_Long"], data["Gal_Lat"], ".", label="Provided")
plt.legend()
plt.xlabel("Galactic longitude (deg)")
plt.ylabel("Galactic latitude (deg)")
plt.show()

coords_icrs = coords_altaz.icrs

ra = coords_icrs.ra.deg
dec = coords_icrs.dec.deg
plt.figure()
plt.plot(ra, dec, ".", markersize=1)
plt.xlabel("Right Ascension (deg)")
plt.ylabel("Declination (deg)")
plt.gca().invert_xaxis()  # standard RA convention
plt.show()

from astropy.coordinates import get_sun

sun_icrs = get_sun(t)
sun_altaz = sun_icrs.transform_to(altaz_frame) 

sun_az = sun_altaz.az.deg
sun_el = sun_altaz.alt.deg 

#What are we observing?
plt.figure()
plt.plot(sun_az, sun_el, ".", markersize=2)
plt.xlabel("Azimuth (deg)")
plt.ylabel("Elevation (deg)")
plt.show()

obs_az = data["Azimuth"]
obs_el = data["Elevation"]

az_offset_calc = obs_az - sun_altaz.az
el_offset_calc = obs_el - sun_altaz.alt

az_offset_calc = az_offset_calc.to(u.deg).value
el_offset_calc = el_offset_calc.to(u.deg).value

#compare with provided offset
az_offset_file = data["Az_Offset"]
el_offset_file = data["El_Offset"]

plt.figure()
plt.plot(az_offset_calc, label="Computed Az offset")
plt.plot(az_offset_file, "--", label="Provided Az offset")
plt.legend()
plt.ylabel("Offset (deg)")
plt.show()

plt.figure()
plt.plot(el_offset_calc, label="Computed El offset")
plt.plot(el_offset_file, "--", label="Provided El offset")
plt.legend()
plt.ylabel("Offset (deg)")
plt.show()

#Plot with residuals
# Residuals
az_residual = az_offset_calc - az_offset_file
el_residual = el_offset_calc - el_offset_file

plt.figure(figsize=(8, 6))

# Top: offsets
plt.subplot(2, 1, 1)
plt.plot(az_offset_calc, label="Computed")
plt.plot(az_offset_file, "--", label="Provided")
plt.ylabel("Azimuth offset (deg)")
plt.legend()
plt.title("Azimuth Pointing Offsets")

# Bottom: residuals
plt.subplot(2, 1, 2)
plt.plot(az_residual)
plt.xlabel("Time sample")
plt.ylabel("Residual (deg)")

plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))

# Top: offsets
plt.subplot(2, 1, 1)
plt.plot(el_offset_calc, label="Computed")
plt.plot(el_offset_file, "--", label="Provided")
plt.ylabel("Elevation offset (deg)")
plt.legend()
plt.title("Elevation Pointing Offsets")

# Bottom: residuals
plt.subplot(2, 1, 2)
plt.plot(el_residual)
plt.xlabel("Time sample")
plt.ylabel("Residual (deg)")

plt.tight_layout()
plt.show()
