from collections import OrderedDict
import numpy as np
import starlord

# Generating some input axes
x = np.linspace(0, 10, 100)
y = np.array([-2., 0., 2., 4., 6.])
z = np.logspace(-1, 1, 15)

# Calculate some outputs (normally these would be some complex model outputs)
# I'm using numpy broadcasting but you can use nested for loops if you prefer
out1 = np.sqrt(x[:, None, None]) + np.sin(y[None, :, None]) / z[None, None, :]
out2 = 2 * x[:, None, None] + y[None, :, None] / z[None, None, :]

starlord.GridGenerator.create_grid(
    grid_name="demo",
    inputs=OrderedDict(x=x, y=y, z=z),
    outputs=dict(out1=out1, out2=out2),
)
