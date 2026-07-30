"""exoklip — direct-imaging exoplanet detection: KLIP / ADI, detection statistics,
contrast curves, and a realistic ADI sequence simulator.

Hard dependencies: numpy, scipy only. astropy (FITS I/O), matplotlib (figures)
and tqdm (progress bars) are optional and imported lazily.
"""

__version__ = "0.1.0"

# Populated during integration — kept import-light on purpose so that a broken
# optional module never prevents importing the rest of the package.
__all__: list[str] = []
