import wedap
import matplotlib.pyplot as plt
import numpy as np

h5_list = ["hk1b.h5", "hk10b.h5", "mab10b.h5", "wevo.h5", "lcas.h5"]
#h5_list = ["west_lowS.h5"]

fig, ax = plt.subplots()

for h5 in h5_list:
    wedap.H5_Plot(h5=h5, data_type="average", first_iter=250, plot_mode="line", 
                  data_label=h5, ax=ax).plot()

    # wedap.H5_Plot(h5=h5, data_type="average", last_iter=1000, plot_mode="line", 
    #               data_label="1000i", ax=ax).plot()
    # wedap.H5_Plot(h5=h5, data_type="average", first_iter=1000, last_iter=2000, plot_mode="line", 
    #               data_label="1-2000i", ax=ax).plot()
    # wedap.H5_Plot(h5=h5, data_type="average", first_iter=2000, plot_mode="line", 
    #               data_label="2-3000i", ax=ax).plot()

def plot_odld(A=2, B=5, C=0.5, x0=1):
    twopi_by_A = 2 * np.pi / A
    half_B = B / 2
    sigma = 0.001 ** (0.5)
    gradfactor = sigma * sigma / 2
    reflect_at = 10

    x = np.arange(0.75,10.1,0.05)
    xarg = twopi_by_A * (x - x0)

    eCx = np.exp(C * x)
    eCx_less_one = eCx - 1.0
    y = half_B / (eCx_less_one * eCx_less_one) * (twopi_by_A * eCx_less_one * np.sin(xarg) + C * eCx * np.cos(xarg))

    #plt.plot(x, np.flip(-np.log(y)), color='k', alpha=0.5, label='ODLD potential', linestyle="--")
    plt.plot(x, y, color='k', alpha=0.5, label='ODLD grad', linestyle="--")
#plot_odld()

def odld_1d_potential(A=2, B=5, C=0.5, x0=1):
    x = np.arange(0.5,10.1,0.05) 
    twopi_by_A = 2 * np.pi / A
    half_B = B / 2

    xarg = twopi_by_A * (x - x0)

    eCx = np.exp(C * x)
    eCx_less_one = eCx - 1.0

    potential = -half_B / eCx_less_one * np.cos(xarg)

    # normalize the plot to have lowest value as baseline
    potential = potential - np.min(potential)

    plt.plot(x, potential, color='k', alpha=0.5, label='ODLD potential', linestyle="--")
    return potential
odld_1d_potential()

plt.legend()
plt.xlabel("ODLD Position")

plt.tight_layout()
#plt.savefig("figures/multi_1d_updated.png", dpi=300, transparent=True)
plt.show()