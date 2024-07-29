
import h5py
import numpy as np
import matplotlib.pyplot as plt

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
        else:
            events.append(0)
    # cumsum the events per iteration
    events = np.cumsum(events)

    return agg_time, events

fig, ax = plt.subplots()

h5_files = ["stdMD", "stdHK", "LCAS", "REAP", "stdHK1D"]
colors = ["gray", "tab:blue", "tab:orange", "tab:red", "tab:green"]

for h5, c in zip(h5_files, colors):
    agg_time, events = calc_events(f"bigdata/west-{h5}.h5")
    ax.plot(agg_time, events, label=h5, color=c)

ax.set_ylabel("Total Recycling Events")
ax.set_xlabel("Aggregate Simulation Time ($\mu$$s$)")
plt.yscale("log")
plt.legend()
plt.show()