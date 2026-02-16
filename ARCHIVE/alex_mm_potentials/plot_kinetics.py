
import wekap
import matplotlib.pyplot as plt

#kfiles = ["stdMD", "stdHK", "LCAS", "REAP"]
#kfiles = ["stdMD", "stdHK", "LCAS"]
kfiles = ["FAST", "TSLC", "LCAS", "REAP"]

colors = ["gray", "tab:blue", "tab:orange", "tab:red"]
#colors = ["gray", "tab:blue", "tab:orange"]

fig, ax = plt.subplots()

for k, c in zip(kfiles, colors):
    kap = wekap.Kinetics(direct=f"ANALYSIS/{k}/direct.h5", ax=ax, label=k,
                         moltime=False, color=c)
    kap.plot_rate()

plt.legend()
plt.xlim(0, 1000)
#plt.xlabel("WE Iteration")
#plt.savefig("Z-well-comp.pdf")
plt.savefig("2D_ODLD_comp.pdf")
#plt.show()
