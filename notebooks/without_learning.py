# Begin Code
from typing import List

import matplotlib  #type: ignore
import matplotlib.pyplot as plt  #type: ignore
# %matplotlib inline
import pandas as pd  #type: ignore

import leabra7 as lb

# Begin Markdown
# # Two Neuron Network

# Begin Markdown
# ### Network Creation

# Begin Code
# Create the network
net = lb.Net()

# Set options for our layers
layer_spec = lb.LayerSpec(
    # For each layer, log the unit potential
    log_on_cycle=("unit_v_m", "unit_act", "unit_i_net", "unit_net",
                  "unit_gc_i", "unit_adapt", "unit_spike"))

# Create our layers and projections
net.new_layer("input", 1, layer_spec)
net.new_layer("output", 1, layer_spec)
net.new_projn("proj1", "input", "output")

# Begin Markdown
# ### Run Network

# Begin Code
# Clamp inputs
net.clamp_layer(name="input", acts=[1])

# Run 200 cycles
for i in range(200):
    net.cycle()

# Begin Markdown
# ### Graphing Logs

# Begin Code
wholeLog, partLog = net.logs(freq="cycle", name="output")
partLog.plot(x='time', figsize=(10, 6))

# Begin Markdown
# # One to Many Neuron Network

# Begin Code
# Define projection spec to have random weights
projn_spec = lb.ProjnSpec(dist=lb.Gaussian(mean=0.5, var=0.3))

# Begin Markdown
# ### Network Creation

# Begin Code
# Create the network
net = lb.Net()

# Create our layers and projections
net.new_layer("input", 1, layer_spec)
net.new_layer("output", 10, layer_spec)
net.new_projn("proj1", "input", "output", spec=projn_spec)

# Begin Markdown
# ### Run Network

# Begin Code
# Clamp inputs
net.clamp_layer(name="input", acts=[1])

# Run 200 cycles
for i in range(100):
    net.cycle()

# Begin Markdown
# ### Graphing

# Begin Code
whole_log, part_log = net.logs(freq="cycle", name="output")

fig, ax = plt.subplots(figsize=(10, 6))
for name, group in part_log.groupby("unit"):
    group.plot(x="time", y="act", ax=ax, label="unit " + str(name))
ax.set_ylabel("Activation")
ax.set_xlabel("Time")

# Begin Markdown
# # Multiple Layers

# Begin Markdown
# ### Network Creation

# Begin Code
# Create the network
net = lb.Net()

# Create our layers and projections
net.new_layer("input", 1, layer_spec)
net.new_layer("middle", 3, layer_spec)
net.new_layer("output", 5, layer_spec)
net.new_projn("proj1", "input", "middle", spec=projn_spec)
net.new_projn("proj2", "middle", "output", spec=projn_spec)
net.new_projn("proj3", "input", "output", spec=projn_spec)

# Begin Markdown
# ### Run Network

# Begin Code
# Clamp inputs
net.clamp_layer(name="input", acts=[1])

# Run 200 cycles
for i in range(100):
    net.cycle()

# Begin Markdown
# ### Logging

# Begin Code
whole_log_in, part_log_in = net.logs(freq="cycle", name="input")
whole_log_mid, part_log_mid = net.logs(freq="cycle", name="middle")
whole_log_out, part_log_out = net.logs(freq="cycle", name="output")

# Begin Markdown
# ### Plotting
#
#


# Begin Code
# Function to plot data for a certain attribute for each unit of layer
def plot_by_unit(axes: List[matplotlib.axes.Axes], log: pd.DataFrame,
                 attr_name: str, title: str, location: int) -> None:
    for unit_name, unit_data in log.groupby("unit"):
        unit_data.plot(
            x="time",
            y=attr_name,
            ax=axes[location],
            title=title,
            label="unit " + str(unit_name))


# Begin Code
fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(10, 20))
attr = "act"
plot_by_unit(
    axes=ax, log=part_log_in, attr_name=attr, title="input", location=0)
plot_by_unit(
    axes=ax, log=part_log_mid, attr_name=attr, title="middle", location=1)
plot_by_unit(
    axes=ax, log=part_log_out, attr_name=attr, title="output", location=2)
