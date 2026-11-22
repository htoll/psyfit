"""Write the two small calibration pickles the tests read.

They have the structure of PsyFit's real spectral calibration files (a
"saving_info_<date>_no_tracking.pkl" grid and a "<date>_fits.pkl"), written by
NumPy so that the pickle stream is what the microscope's analysis produces,
but every number is made up:

    object point  (x, y) = (20 + 15 i, 30 + 10 j)        i in 0..5, j in 0..3
    spectral point        = (1.03 x + 190, 1.01 y + 250)
    wavelength fit        = (56 + 0.1 i, 0.022, 376)      for points with even i

    python make_fixtures.py        (needs numpy; run from this folder)
"""
import pickle
import numpy as np

grid, fits = {}, {}
n = 0
for j in range(4):
    for i in range(6):
        pid = f"P{n:04d}"
        x, y = np.float64(20 + 15 * i), np.float64(30 + 10 * j)
        spot = np.array([30000.0 + n, x, y, 3.0, 2.5, 0.0, 24000.0])
        grid[pid] = ["sim", (x, y), spot, (np.float64(1.03 * x + 190), np.float64(1.01 * y + 250)), np.arange(100, dtype=np.float64) + n]
        if i % 2 == 0:
            fits[pid] = (np.float64(56 + 0.1 * i), np.float64(0.022), np.float64(376.0))
        n += 1

with open("saving_info_990101_no_tracking.pkl", "wb") as f:
    pickle.dump(grid, f)
with open("990101_fits.pkl", "wb") as f:
    pickle.dump(fits, f)
print(len(grid), "grid points,", len(fits), "fits")
