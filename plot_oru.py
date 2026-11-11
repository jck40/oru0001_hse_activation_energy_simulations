import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fit_exp import compare, EXP, G
a = 2.71
runs = [('oru_renisch', 'Renisch + V(a)=265 meV, symmetric rate rule', 'k', '-'),
        ('var_lam0', 'same potential, initial-state rate rule', 'C0', '--'),
        ('var_stampfl', 'DFT pairs (Stampfl): 265, 44, −25 meV', 'C2', '-.'),
        ('var_noattr', 'no attraction: 265, 30, 0, 0 meV', 'C1', ':'),
        ('var_strongattr', 'attraction ×1.35: 265, 30, −65, −35 meV', 'C3', (0, (5, 1, 1, 1)))]
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8), gridspec_kw=dict(width_ratios=[1.7, 1]))
for pre, lab, c, ls in runs:
    off, chi2, dk, Erel, Ee = compare(pre, verbose=False)
    keep = dk > 0.15                                   # drop the two noisiest smallest-dK points
    x = np.concatenate([dk[keep], (G - dk[keep])[::-1]])
    y = np.concatenate([Erel[keep], Erel[keep][::-1]]) + off
    e = np.concatenate([Ee[keep], Ee[keep][::-1]])
    ax.plot(x, y, color=c, ls=ls, lw=2 if pre == 'oru_renisch' else 1.4, label=f'{lab}  (χ²={chi2:.1f})')
    if pre == 'oru_renisch':
        ax.fill_between(x, y - e, y + e, color='0.6', alpha=0.4, lw=0)
ax.errorbar(EXP[:, 0], EXP[:, 1], EXP[:, 2], fmt='o', color='C9', mec='k', ms=7, capsize=3, zorder=10,
            label='HeSE, 0.06 ML, ~850 K')
ax.axvline(G / 2, color='k', ls='--', lw=0.8)
ax.text(G / 2 + 0.02, 352, 'M', fontsize=11)
ax.set_xlim(0, G); ax.set_ylim(340, 720)
ax.set_xlabel(r'$\Delta K$ ($\AA^{-1}$), $\Gamma$–M–$\Gamma^\prime$')
ax.set_ylabel(r'$E_{\rm eff}$ (meV)')
ax.set_title(r'O/Ru(0001): KMC $E_{\rm eff}(\Delta K)$, one fitted offset per curve (7 dof)')
ax.legend(fontsize=7.8, loc='upper left')
r = np.array([1, np.sqrt(3), 2, np.sqrt(7), 3])
pots = {'Renisch + V(a)': [265, 30, -48, -25, 0], 'DFT pairs (Stampfl)': [265, 44, -25, 0, 0],
        'no attraction': [265, 30, 0, 0, 0], 'attraction ×1.35': [265, 30, -65, -35, 0]}
for (k, v), c in zip(pots.items(), ['k', 'C2', 'C1', 'C3']):
    ax2.plot(r, v, 'o-', color=c, label=k, ms=5)
ren_r = [np.sqrt(3), np.sqrt(3), 2, 2, np.sqrt(7), 3]; ren_v = [29, 27, -45, -52, -25, 0]
ax2.plot(ren_r, ren_v, 's', mfc='none', color='C4', ms=9, label='Renisch et al. Fig. 5 (read off)')
ax2.axhline(0, color='0.5', lw=0.8)
ax2.set_ylim(-80, 80)
ax2.annotate('V(a) = 265 meV\n(off scale)', xy=(1.0, 78), xytext=(1.25, 55), fontsize=8,
             arrowprops=dict(arrowstyle='->'))
ax2.set_xlabel(r'O–O distance ($a$ = 2.71 Å)')
ax2.set_ylabel('V (meV)')
ax2.set_title('pair potentials used (hcp–hcp)')
ax2.legend(fontsize=7.8)
fig.tight_layout()
fig.savefig('ORu_Eeff_vs_experiment.png', dpi=150)
print('saved')
