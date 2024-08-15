
import numpy as np
import matplotlib.pyplot as plt

#plt.style.use("/home/dty7/Projects/wedap/wedap/styles/default.mplstyle")
plt.style.use("../werl.mplstyle")

#h5_files = ["stdMD", "stdHK1D", "stdHK2D", "LCAS", "REAP", "FAST", "TSLC"]
#colors = ["gray", "tab:blue", "tab:orange", "tab:red", "tab:green"]
#colors = ["k", "dimgray", "tab:blue", "tab:orange"]
colors = ["tab:blue", "tab:orange", "tab:red", "tab:green", "tab:pink", "tab:purple"]
#colors = ["tab:blue", "tab:orange", "tab:red", "tab:green"]
pcoords = ["Orientation Angle 1 ($\degree$)", "Orientation Angle 2 ($\degree$)", "T188-T188 Distance ($\AA$)", "C$_2$ Angle ($\degree$)", "M1 W184 $\chi_1$ ($\degree$)", "M2 W184 $\chi_1$ ($\degree$)"]
pcoords = ["Orientation\nAngle 1 ($\degree$)", "Orientation\nAngle 2 ($\degree$)", "T188-T188\nDistance ($\AA$)", "C$_2$ Angle ($\degree$)"]

#fig, ax = plt.subplots(figsize=(7.5,4))
fig, ax = plt.subplots(figsize=(6,4))

# grab all weights 
agg_weights = []
for j in range(5):
    #weights = np.loadtxt(f"template-REAP-noscale/0{j}/op_weights_v0{j}.txt")
    weights = np.loadtxt(f"template-REAP/0{j}/op_weights.txt")
    agg_weights.append(weights)

# WE iterations
we_iters = [t for t in range(weights.shape[0])]
# dimensionality of pcoord
n_pcoords = weights.shape[1]

# calc stats
avg_w = np.average(agg_weights, axis=0)
std_w = np.std(agg_weights, axis=0)
# plot
for i in range(n_pcoords):
    ax.plot(we_iters, avg_w[:,i], label=pcoords[i], color=colors[i])
    ax.fill_between(we_iters, avg_w[:,i] - std_w[:,i], avg_w[:,i] + std_w[:,i], color=colors[i], alpha=0.2)

# plot
#avg_events = np.sum(agg_events, axis=0)
#std_events = np.std(agg_events, axis=0)
#ax.plot(agg_time, avg_events, label=labels[i], color=colors[i], linewidth=2)
#ax.fill_between(agg_time, avg_events - std_events, avg_events + std_events, color=c, alpha=0.2)

ax.set_xlabel("WE Iteration")
ax.set_ylabel("Progress Coordinate Weights")
plt.ylim(0,1)
plt.legend(frameon=False, loc="center left", bbox_to_anchor=(1,0.5), labelspacing=2)
plt.tight_layout()
plt.savefig("figs/weights2.pdf", bbox_inches='tight')
plt.show()
