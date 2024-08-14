
import wekap
import matplotlib.pyplot as plt
import numpy as np

#plt.style.use("/home/dty7/Projects/wedap/wedap/styles/default.mplstyle")
plt.style.use("default.style")

#kfiles = ["stdHK1D", "stdHK2D", "LCAS", "REAP", "FAST", "TSLC"]
kfiles = ["stdHK1D", "stdHK2D", "LCAS", "REAP"]
#kfiles = ["stdMD", "stdHK", "LCAS"]
#colors = ["tab:blue", "tab:orange", "tab:red", "tab:green", "tab:pink", "tab:purple"]
#colors = ["tab:blue", "tab:orange", "tab:red", "tab:green"]
colors = ["silver", "dimgray", "tab:blue", "tab:orange"]
#colors = ["gray", "tab:blue", "tab:orange"]
labels = ["HK-1D", "HK-2D", "LCAS", "REAP"]

fig, ax = plt.subplots()

# get stdMD reference rate from 5 10k trials
std_dh5s = [f"template-stdMD/{i}/ANALYSIS/REPLACEWITHVERSION/direct.h5" for i in range(5)]
kap = wekap.Kinetics(direct=std_dh5s[0], ax=ax, color='k',
                     x_units="agg", tau=21e-12)
std_x, rates, avgs, crs = kap.plot_multi_rates(std_dh5s, plotting=False)
#std_x, rates, avgs, crs = kap.plot_multi_rates(std_dh5s, plotting=True)
#print(f"stdMD avg: {avgs[-1]}
ax.axhline(avgs[-1], color='k', linestyle='--', label='cMD')
#ax.axhline(avgs[-1], color='k', linestyle='--', label='cMD')
#ax.axhline(24374526.25947432, color='k', linestyle='--', label='cMD', linewidth=2)
#print(f"stdMD_cr_width: {crs[-1][1] - crs[-1][0]:.5E}")
#import sys; sys.exit(0)
#print(f"stdMD_cr_width: {crs:.5E}")
#print(f"stdMD_deltaKsq: {2640948958016.799:.5E}")

#ax.fill_between(std_x, crs[-1][0], crs[-1][1], alpha=0.8, color='k')
#print(np.std(rates, axis=0).shape)
#print(np.std(rates, axis=1).shape)
#print(f"t(BF): {std_x[-1]}")
#print(f"stderr k(BF)^2: {(np.std(rates, axis=0)[-1]/np.sqrt(5))**2}")

# Calculation of efficiency compared to brute force
# equation is: t(BF) / t(WE) * (deltaK(BF) / deltaL(WE))^2
# TODO: detect proximity to bf within error and select t_we/avg_we/cr_width_we from there
def calc_efficiency(t_we, avg_rate_we, cr_width_we, 
                    t_bf=16.8, avg_rate_bf=2.43745E+07, cr_width_bf=5.56542E+06):
    t_ratio = t_bf / t_we
    # width of confidence / mean rate : bf / we
    deltaK_ratio = (cr_width_bf / avg_rate_bf) / (cr_width_we / avg_rate_we)
    S = t_ratio * (deltaK_ratio)**2
    return S

# then plot the average of each resampler (3 trials each)
S_calcs = []
trials = 10
for i, k in enumerate(kfiles):
    dh5s = [f"template-{k}/{i}/ANALYSIS/REPLACEWITHVERSION/direct.h5" for i in range(trials)]
    kap = wekap.Kinetics(direct=dh5s[0], ax=ax, color=colors[i], label=labels[i],
                         x_units="agg", tau=21e-12)
    std_x, rates, avgs, crs = kap.plot_multi_rates(dh5s, plotting=False)
    ax.fill_between(std_x, crs[:,0], crs[:,1], alpha=0.25, color=colors[i])
    ax.plot(std_x, avgs, label=labels[i], color=colors[i], linewidth=2)
    print(std_x)

    #std_x, rates, avgs, crs = kap.plot_multi_rates(dh5s, plotting=False)
    # print S (efficiency)
    #deltaKsq = (np.std(rates, axis=0)[-1]/np.sqrt(trials))**2
    #S_calc = calc_efficiency(std_x[-1], avgs[-1], crs[-1][1] - crs[-1][0])
    #print(f"avg_rate_{k}: {avgs[-1]:.5E}")
    #print(f"crs_{k}: {crs[-1][1]-crs[-1][0]:.5E}")
    #print(f"stderr_rate_{k}: {deltaKsq:.5E}")
    #print(f"S_{k}: {S_calc}")
    #S_calcs.append(S_calc)

#print(dict(zip(kfiles,S_calcs)))

ax.set_ylabel("Rate Estimate ($s^{-1}$)")
ax.set_xlabel("Aggregate Simulation Time ($steps \cdot 10^6$)")

plt.yscale("log")
plt.legend(frameon=False)
#plt.xlim(0,2.8)
plt.tight_layout()
plt.savefig("figs/kinetics-leg.pdf")
plt.show()


