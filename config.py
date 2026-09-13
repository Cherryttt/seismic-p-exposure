from __future__ import annotations

import os

# ---- Mainshock seed (used to find/confirm mainshock via USGS) ----

# Lushan earthquake occurred 2013-04-20 local morning; in UTC around 00:02:46Z.
# We still query a 2-day window and pick the largest magnitude in a small radius.
MAINSHOCK_SEARCH_START_UTC = "2013-04-19T00:00:00Z"
MAINSHOCK_SEARCH_END_UTC = "2013-04-21T00:00:00Z"
MAINSHOCK_MINMAG = 6.0

# Initial guess for mainshock location (Lushan/Yaan, Sichuan)
MAINSHOCK_LAT = 30.30
MAINSHOCK_LON = 102.99
MAINSHOCK_SEARCH_RADIUS_KM = 300

# ---- Aftershock query window ----
#
# "Big data" settings (target 100k+ total events in batch mode):
# - longer window
# - larger radius
# - lower min magnitude
#
# If USGS/your network is slow, start with DAYS_AFTER=180 and MINMAG_DOWNLOAD=0.5.
RADIUS_KM = 400
DAYS_AFTER = 365

# Magnitude cutoff for download (keep low; we will apply Mc later)
MINMAG_DOWNLOAD = 0.5

# ---- Modeling / sensitivity ----

# Depth grouping (km)
DEPTH_BINS_KM = [0, 10, 20, 40, 80, 300]

# Simple "regional" splits to mimic fault-segment differences (fast to do in 2 days).
# Each region is (name, predicate) where predicate gets (lat, lon) and returns bool.
REGIONS = [
    ("West_of_lon0", lambda lat, lon: lon < MAINSHOCK_LON),
    ("East_of_lon0", lambda lat, lon: lon >= MAINSHOCK_LON),
    ("South_of_lat0", lambda lat, lon: lat < MAINSHOCK_LAT),
    ("North_of_lat0", lambda lat, lon: lat >= MAINSHOCK_LAT),
]

# Fit start-time (days) sensitivity grid.
TMIN_GRID_DAYS = [0.01, 0.05, 0.1]  # ~15min, 1.2h, 2.4h

# Magnitude completeness sensitivity grid (use fixed Mcs; avoids debate about Mc estimator)
MC_GRID = [1.5, 2.0, 2.5, 3.0]

# Bootstrap resamples for p-value CI (keep moderate for speed)
BOOTSTRAP_N = 200

# Random seed for reproducibility
SEED = 7

# ---- Exposure (population) settings ----

# Buffers around each mainshock for exposure calculation (km)
EXPOSURE_BUFFER_KM = [10, 25, 50]

# Target population raster path. The raster is intentionally not committed.
# Example (PowerShell): $env:POP_RASTER_PATH = "D:\data\population.tif"
POP_RASTER_PATH = os.environ.get("POP_RASTER_PATH", "")

# Distance rings (km) for subgroup expansion within each mainshock.
# Example: [0, 50, 150, 400] creates rings 0-50, 50-150, 150-400 km.
DISTANCE_RINGS_KM = [0, 50, 150, 400]

# ---- Multi-mainshock (Sichuan-Yunnan) batch mode ----

# Rough bounding box for Sichuan-Yunnan region
# (You can adjust later to be tighter around Sichuan-Yunnan seismic belts.)
REGION_MIN_LAT = 22.0
REGION_MAX_LAT = 35.0
REGION_MIN_LON = 97.5
REGION_MAX_LON = 105.5

# How many mainshocks to compare (sorted by time asc)
MAX_MAINSHOCKS = 30

# Minimum magnitude to consider as a mainshock candidate in the region.
MAINSHOCK_BATCH_MINMAG = 5.5

# Decluster approximation: skip candidates occurring within this many days
# of an already-selected mainshock (prevents picking aftershocks as mainshocks).
DECLUSTER_DAYS = 30

# Batch mainshock search time range (UTC)
MAINSHOCK_BATCH_START_UTC = "2000-01-01T00:00:00Z"
MAINSHOCK_BATCH_END_UTC = "2025-12-31T23:59:59Z"
