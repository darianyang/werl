
import wekap
import matplotlib.pyplot as plt

plt.style.use("/Users/darian/github/wedap/wedap/styles/default.mplstyle")

kfiles = ["stdMD", "stdHK", "LCAS", "REAP", "stdHK1D"]
#kfiles = ["stdMD", "stdHK", "LCAS"]
colors = ["gray", "tab:blue", "tab:orange", "tab:red", "tab:green"]
#colors = ["gray", "tab:blue", "tab:orange"]

fig, ax = plt.subplots(figsize=(10,4))

for k, c in zip(kfiles, colors):
    kap = wekap.Kinetics(direct=f"ANALYSIS-BD2/{k}/direct.h5", ax=ax, label=k, 
                         x_units="agg", color=c, tau=21e-12)
    kap.plot_rate()

plt.xlim(0,5)

plt.legend()
#plt.xlim(0,5000)

plt.tight_layout()
#plt.savefig("Z-well-comp.pdf")
#plt.savefig("Z-well-B-comp.pdf")
plt.show()
plt.savefig("figures/kinetics2.pdf")
