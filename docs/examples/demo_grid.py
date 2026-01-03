from collections import OrderedDict
import numpy as np
import starlord

# Generating some input axes
x = np.linspace(0, 10, 100)
y = np.logspace(-1, 1., 20)

# Calculate some outputs (normally these would be some complex model outputs)
# I'm using numpy broadcasting but you can use nested for loops if you prefer
out1 = np.sqrt(x[:, None]) + y[None, :]**2
out2 = 2 * x[:, None]**3 + np.sqrt(y[None, :])

starlord.GridGenerator.create_grid(
    grid_name="demo_grid",
    inputs=OrderedDict(x=x, y=y),
    outputs=dict(out1=out1, out2=out2),
    derived=dict(ratio="demo_grid.out1 / demo_grid.out2"),
)
