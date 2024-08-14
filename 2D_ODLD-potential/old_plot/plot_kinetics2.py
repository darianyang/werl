import wekap
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

kfiles = ["stdHK1D", "stdHK2D", "LCAS", "REAP", "FAST", "TSLC"]
#kfiles = ["stdMD", "stdHK", "LCAS"]
colors = ["tab:blue", "tab:orange", "tab:red", "tab:green", "tab:pink", "tab:purple"]
#colors = ["gray", "tab:blue", "tab:orange"]

fig, ax = plt.subplots()

# get stdMD reference rate from 10 1k trials
std_dh5s = [f"template-stdMD/{i}/ANALYSIS/REPLACEWITHVERSION/direct.h5" for i in range(10)]
kap = wekap.Kinetics(direct=std_dh5s[0], ax=ax, color='k',
                     x_units="agg", tau=21e-12)
std_x, rates, avgs, crs = kap.plot_multi_rates(std_dh5s, plotting=False)
#std_x, rates, avgs, crs = kap.plot_multi_rates(std_dh5s, plotting=True)
ax.axhline(avgs[-1], color='k', linestyle='--', label='stdMD')
ax.fill_between([10*x for x in std_x], crs[:,0], crs[:,1], alpha=0.25, color='k')
ax.plot([10*x for x in std_x], avgs, c='k')

# then plot the average of each resampler (10 trials each)
for k, c in zip(kfiles, colors):
    print(k)
    dh5s = [f"template-{k}/{i}/ANALYSIS/REPLACEWITHVERSION/direct.h5" for i in range(10)]
    kap = wekap.Kinetics(direct=dh5s[0], ax=ax, color=c, label=k,
                         x_units="agg", tau=21e-12)
    std_x, rates, avgs, crs = kap.plot_multi_rates(dh5s)

plt.legend()
plt.xlim(0,1000)
plt.show()
