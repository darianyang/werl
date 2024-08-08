
import wekap
import matplotlib.pyplot as plt

kfiles = ["stdHK1D", "stdHK2D", "LCAS", "REAP", "FAST", "TSLC"]
#kfiles = ["stdMD", "stdHK", "LCAS"]
colors = ["tab:blue", "tab:orange", "tab:red", "tab:green", "tab:pink", "tab:purple"]
#colors = ["gray", "tab:blue", "tab:orange"]

fig, ax = plt.subplots()

# get stdMD reference rate from 5 10k trials
std_dh5s = [f"template-stdMD/0{i}/ANALYSIS/v0{i}/direct.h5" for i in range(5)]
kap = wekap.Kinetics(direct=std_dh5s[0], ax=ax, color='k',
                     x_units="agg", tau=21e-12)
#std_x, rates, avgs, crs = kap.plot_multi_rates(std_dh5s, plotting=False)
std_x, rates, avgs, crs = kap.plot_multi_rates(std_dh5s, plotting=True)
ax.axhline(avgs[-1], color='k', linestyle='--', label='stdMD')
#ax.fill_between(std_x, crs[-1][0], crs[-1][1], alpha=0.8, color='k')

# then plot the average of each resampler (3 trials each)
for k, c in zip(kfiles, colors):
    dh5s = [f"template-{k}/0{i}/ANALYSIS/v0{i}/direct.h5" for i in range(3)]
    kap = wekap.Kinetics(direct=dh5s[0], ax=ax, color=c, label=k,
                         x_units="agg", tau=21e-12)
    std_x, rates, avgs, crs = kap.plot_multi_rates(dh5s)

#for k, c in zip(kfiles, colors):
#    kap = wekap.Kinetics(direct=f"ANALYSIS-BD2/{k}/direct.h5", ax=ax, label=k, 
#                         x_units="agg", color=c, tau=21e-12)
#    kap.plot_rate()

plt.legend()
#plt.xlim(0,5000)
plt.show()
