
import wekap
import matplotlib.pyplot as plt

kfiles = ["stdMD", "stdHK", "LCAS", "REAP"]

fig, ax = plt.subplots()

for k in kfiles:
    kap = wekap.Kinetics(direct=f"ANALYSIS/{k}/direct.h5", ax=ax, label=k, moltime=False)
    kap.plot_rate()

plt.legend()
plt.show()