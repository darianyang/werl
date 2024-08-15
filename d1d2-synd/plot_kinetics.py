
import wekap
import matplotlib.pyplot as plt
import numpy as np

#plt.style.use("/home/dty7/Projects/wedap/wedap/styles/default.mplstyle")
plt.style.use("../werl.mplstyle")

#kfiles = ["stdHK1D", "stdHK2D", "LCAS", "REAP", "FAST", "TSLC"]
#kfiles = ["stdMD", "stdHK", "LCAS", "REAP"]
kfiles = ["stdHK", "LCAS", "REAP"]
#kfiles = ["stdMD", "REAP-noscale", "REAP-noscale2", "REAP"]
#kfiles = ["stdMD", "stdHK", "REAP-noscale", "REAP-noscale2"]
#kfiles = ["stdMD", "stdHK", "LCAS"]
#colors = ["tab:blue", "tab:orange", "tab:red", "tab:green", "tab:pink", "tab:purple"]
#colors = ["tab:blue", "tab:orange", "tab:red", "tab:green"]
colors = ["silver", "dimgray", "tab:blue", "tab:orange"]
colors = ["tab:gray", "tab:red", "tab:blue", "tab:orange"]
colors = ["tab:gray", "tab:blue", "tab:orange", "tab:green"]
colors = ["tab:gray", "tab:blue", "tab:orange"]
#colors = ["tab:blue", "tab:orange", "tab:green"]
#colors = ["gray", "tab:blue", "tab:orange"]
#labels = ["No WE", "WE", "WE-AS", "WE-RL"]
labels = ["WE", "WE-AS", "WE-RL"]

fig, ax = plt.subplots()

# get stdMD reference rate from 5 trials
#std_dh5s = [f"template-stdMD/0{i}/ANALYSIS/v0{i}/direct.h5" for i in range(5)]
#kap = wekap.Kinetics(direct=std_dh5s[0], ax=ax, color='k',
#                     x_units="agg", tau=21e-12)
#std_x, rates, avgs, crs = kap.plot_multi_rates(std_dh5s, plotting=False)
#std_x, rates, avgs, crs = kap.plot_multi_rates(std_dh5s, plotting=True)
#print(f"stdMD avg: {avgs[-1]}")
#ax.axhline(avgs[-1], color='k', linestyle='--', label='stdMD')
#ax.axhline(24374526.25947432, color='k', linestyle='--', label='cMD', linewidth=2)
ax.axhline(261190, color='k', linestyle='--', label='MSM', linewidth=2)
#print(f"stdMD_rate: {24374526.25947432:.5E}")
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
trials = 5
for i, k in enumerate(kfiles):
    if k == "stdMD":
        dh5s = [f"template-{k}/0{i}/ANALYSIS/REPLACEWITHVERSION/direct.h5" for i in range(trials)]
        #dh5s = [f"template-{k}/0{i}/ANALYSIS/v0{i}/direct.h5" for i in range(trials)]
    else:
        dh5s = [f"template-{k}/0{i}/ANALYSIS/v0{i}/direct.h5" for i in range(trials)]
    kap = wekap.Kinetics(direct=dh5s[0], ax=ax, color=colors[i], label=labels[i],
                         x_units="agg", tau=200e-12)
    std_x, rates, avgs, crs = kap.plot_multi_rates(dh5s, plotting=False)
    if k == "stdMD":
        #ax.plot(std_x[::10]*10, avgs, label=labels[i], color=colors[i], linewidth=2)
        #ax.fill_between(std_x[::10]*10, crs[:,0], crs[:,1], alpha=0.25, color=colors[i])
        ax.plot(std_x+1, avgs, label=labels[i], color=colors[i], linewidth=2, linestyle="--")
        ax.fill_between(std_x+1, crs[:,0], crs[:,1], alpha=0.25, color=colors[i])
    else:
        ax.plot(std_x, avgs, label=labels[i], color=colors[i], linewidth=2)
        ax.fill_between(std_x, crs[:,0], crs[:,1], alpha=0.25, color=colors[i])

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
#ax.set_xlabel("Aggregate Simulation Time ($steps \cdot 10^6$)")
ax.set_xlabel("Aggregate Simulation Time ($\mu$$s$)")
#for k, c in zip(kfiles, colors):
#    kap = wekap.Kinetics(direct=f"ANALYSIS-BD2/{k}/direct.h5", ax=ax, label=k, 
#                         x_units="agg", color=c, tau=21e-12)
#    kap.plot_rate()

plt.yscale("log")
plt.legend(frameon=False)
#plt.xlim(-0.1,2.33)
#plt.xlim(-1,22.3)
plt.xlim(-1,15.3)
#plt.xlim(-1,12.3)
plt.tight_layout()
#plt.savefig("figs/kinetics-noleg.pdf")
#plt.savefig("figs/kinetics-leg.pdf")
plt.savefig("figs/kinetics4.pdf")
plt.show()


