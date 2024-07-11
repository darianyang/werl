
import wekap
import matplotlib.pyplot as plt

kfiles = ["stdMD", "stdHK", "LCAS", "REAP"]
kfiles = ["stdMD", "stdHK", "stdHK-opt"]
colors = ["gray", "tab:blue", "tab:orange", "tab:red"]
colors = ["gray", "tab:blue", "tab:orange"]

fig, ax = plt.subplots()

for k, c in zip(kfiles, colors):
    kap = wekap.Kinetics(direct=f"ANALYSIS-BD/{k}/direct.h5", ax=ax, label=k, 
                         x_units="agg", color=c, tau=21e-12)
    kap.plot_rate()

plt.legend()
#plt.xlim(0,5000)
#plt.savefig("Z-well-comp.pdf")
#plt.savefig("Z-well-B-comp.pdf")
plt.show()
