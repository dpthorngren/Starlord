from collections import OrderedDict
import numpy as np
import pandas as pd
import starlord

grid = pd.read_csv("hotJupiters.csv", skipinitialspace=True, sep=",")
print(grid.columns)

# Rearrange the grid into 3 outputs in 5 dimensions
inputs, outputs = starlord.GridGenerator.restructure_grid(grid.values, (0, 1, 2, 3, 4), (5, 6, 7))
(mass, zpl, flux, heating, age) = inputs
radius, entropy, luminosity = outputs

# The axes of the grids -- order matters!
# Mass, flux, and age span multiple order of magnitude so it's best to interpolate in logspace
inputs = OrderedDict(
    log_mass=np.log10(mass),
    zpl=zpl,
    log_flux=np.log10(flux),
    heating=heating,
    log_age=np.log10(age),
)

# The output values to interpolate on, order does not matter anymore.
outputs = dict(
    log_radius=np.log10(radius),
    entropy=entropy,
    log_luminosity=np.log10(luminosity),
)

# Optionally, derived values that can be calculated from the outputs or inputs
# Note that adding or removing "log_*" is understood without an entry here
derived = dict(
    tint="math.pow(g.hotJupiters.luminosity / (7.125593e-4 * (g.hotJupiters.radius * 6.9911e9)**2), 0.25)",
    typical_heating="0.0237 * math.exp(-(g.hotJupiters.log_flux - 9.14)**2 / (2 * .37**2))",
)

# Input everything into the create_grid function
starlord.GridGenerator.create_grid(
    "hotJupiters",
    inputs=inputs,
    outputs=outputs,
    derived=derived,
    citations="Thorngren & Fortney (2018; 10.3847/1538-3881/aaba13)",
    version="2",
    notes="Appropriate for fitting the compositions of hot Jupiters and cool giants (T_eq < 1000 K).  For the latter," +
    " set heating to 0.",
)
