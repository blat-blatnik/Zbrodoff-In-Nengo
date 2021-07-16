import matplotlib.pyplot as plt

plt.rc('font', size=16)
plt.rc('legend', fontsize=11)
plt.xlabel('Addend')
plt.ylabel('Response time (s)')
plt.plot([2, 3, 4], [1.840, 2.461, 2.815], label='Human counting', c='tab:blue',   ls='-',  marker='.', ms=20)
plt.plot([2, 3, 4], [1.136, 1.212, 1.167], label='Human recall',   c='tab:blue',   ls='--', marker='.', ms=20)
plt.plot([2, 3, 4], [1.976, 2.340, 2.687], label='ACT-R counting', c='tab:orange', ls='-',  marker='.', ms=20)
plt.plot([2, 3, 4], [1.319, 1.360, 1.389], label='ACT-R recall',   c='tab:orange', ls='--', marker='.', ms=20)
plt.plot([2, 3, 4], [2.250, 2.750, 3.250], label='Nengo counting', c='tab:green',  ls='-',  marker='.', ms=20)
plt.plot([2, 3, 4], [1.250, 1.250, 1.250], label='Nengo recall',   c='tab:green',  ls='--', marker='.', ms=20)
plt.xlim(1.5, 4.5)
plt.ylim(0.5, 4.0)
plt.xticks([2, 3, 4])
plt.yticks([1, 2, 3])
plt.legend()
plt.show()