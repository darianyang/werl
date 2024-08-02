
import wekap
import matplotlib.pyplot as plt

#kfiles = ["stdMD", "stdHK", "LCAS", "REAP", "stdHK1D"]
#kfiles = ["stdMD", "stdHK", "LCAS"]
kfiles = ["stdMD", "LCAS", "REAP", "TSLC", "FAST", "stdHK"]
colors = ["gray", "tab:blue", "tab:orange", "tab:red", "tab:green", "tab:purple"]
#colors = ["gray", "tab:blue", "tab:orange"]

fig, ax = plt.subplots()

for k, c in zip(kfiles, colors):
    kap = wekap.Kinetics(direct=f"ANALYSIS/{k}/direct.h5", ax=ax, label=k, 
                         x_units="agg", color=c, tau=21e-12)
    kap.plot_rate()

handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))

#plt.legend(by_label.values(), by_label.keys(), loc='lower right')
plt.legend(by_label.values(), by_label.keys(), loc='lower right')
#plt.xlim(0,5000)
#plt.savefig("2D_ODLD-comp.pdf")
plt.show()
