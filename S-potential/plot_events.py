
import h5py
import numpy as np
import matplotlib.pyplot as plt

#plt.style.use("/home/dty7/Projects/wedap/wedap/styles/default.mplstyle")
plt.style.use("../werl.mplstyle")

def calc_events(h5, tau=21e-12):
    """
    Get the cumulative sum of events and aggregate simulation times.

    Parameters
    ----------
    h5 : str
        west.h5 file path.
    tau : float
        dynamics time interval between resampling steps
    
    Returns
    -------
    events : array
    agg_time : array
    """
    # open input west.h5 file
    west = h5py.File(h5, "r")

    # X-axis data
    # calc the agg sim time (nsegs * tau)
    n_particles = west["summary"]["n_particles"]
    # cumsum and convert from per second to per microsecond
    agg_time = np.cumsum(tau * n_particles) * 1e6

    # Y-axis data
    # calc n events per iteration
    events = []
    for iter in range(1, west.attrs["west_current_iteration"] + 1):
        # if the new_weights group exists in the h5 file
        if f"iterations/iter_{iter:08d}/new_weights" in west:
            # doesn't have to be prev_seg_id, but using it for length here
            prev_segs = west[f"iterations/iter_{iter:08d}/new_weights/index"]["prev_seg_id"]
            # append the sum of n_segs recycled
            events.append(len(prev_segs))
        # otherwise: 0 events for the iteration
        else:
            events.append(0)
    # cumsum the events per iteration
    events = np.cumsum(events)

    return agg_time, events

fig, ax = plt.subplots(figsize=(6,4))

h5_files = ["stdMD", "stdHK1D", "stdHK2D", "LCAS", "REAP"]
#h5_files = ["stdMD", "stdHK1D", "stdHK2D", "LCAS", "REAP", "FAST", "TSLC"]
colors = ["gray", "tab:blue", "tab:orange", "tab:red", "tab:green"]
#colors = ["gray", "tab:blue", "tab:orange", "tab:red", "tab:green", "tab:pink", "tab:purple"]
colors = ["k", "gray", "tab:gray", "tab:blue", "tab:orange"]
labels = ["Std", "WE-1D", "WE-2D", "WE-AS", "WE-RL"]

for i, h5 in enumerate(h5_files):
    # grab avg and std
    agg_events = []
    for j in range(3):
        agg_time, events = calc_events(f"template-{h5}/0{j}/west-v0{j}.h5")
        agg_events.append(events)
    # plot
    #avg_events = np.average(agg_events, axis=0)
    #std_events = np.std(agg_events, axis=0)
    #ax.plot(agg_time, avg_events, label=h5, color=c)
    #ax.fill_between(agg_time, avg_events - std_events, avg_events + std_events, color=c, alpha=0.2)
    # plot
    avg_events = np.sum(agg_events, axis=0)
    #std_events = np.std(agg_events, axis=0)
    if h5 == "stdMD" or h5 == "stdHK1D":
        ax.plot(agg_time, avg_events, label=labels[i], color=colors[i], linewidth=2, linestyle="--")
    else:
        ax.plot(agg_time, avg_events, label=labels[i], color=colors[i], linewidth=2)
    #ax.fill_between(agg_time, avg_events - std_events, avg_events + std_events, color=c, alpha=0.2)

ax.set_ylabel("Total Successful Events")
#ax.set_xlabel("Aggregate Simulation Time ($\mu$$s$)")
ax.set_xlabel("Aggregate Simulation Time ($steps \cdot 10^6$)")
plt.yscale("log")
#plt.xlim(0,2.8)
plt.xlim(-0.1,1.5)
#plt.legend(frameon=False, loc="center", bbox_to_anchor=(1.2,0.5))
#plt.legend(frameon=False)
plt.tight_layout()
plt.savefig("figs/events4.pdf")
plt.show()
