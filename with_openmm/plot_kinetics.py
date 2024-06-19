
import wekap
import matplotlib.pyplot as plt

kfiles = ["stdMD", "stdHK", "LCAS", "REAP"]
#kfiles = ["stdMD", "stdHK", "LCAS"]
colors = ["gray", "tab:blue", "tab:orange", "tab:red"]
#colors = ["gray", "tab:blue", "tab:orange"]

fig, ax = plt.subplots()

for k, c in zip(kfiles, colors):
    kap = wekap.Kinetics(direct=f"ANALYSIS-B/{k}/direct.h5", ax=ax, label=k, moltime=False, color=c)
    kap.plot_rate()

plt.legend()
#plt.savefig("Z-well-comp.pdf")
plt.show()