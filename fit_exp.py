import numpy as np, sys
a = 2.71
G = 4*np.pi/(np.sqrt(3)*a)                     # Gamma-Gamma' along Gamma-M
EXP = np.array([[0.40,439.33,25.89],[0.50,480.35,20.79],[0.90,489.61,7.20],[1.00,556.76,36.24],
                [1.34,565.00,19.51],[1.34,523.62,18.17],[1.80,493.58,36.00],[2.20,390.51,33.30]])
def compare(prefix, verbose=True):
    r = np.load(prefix + '_results.npz')
    m = r['az'] == '<11-2> (G-M)'
    dk, E, Ee = r['dk'][m], 1e3*r['E_alpha'][m], 1e3*r['E_alpha_err'][m]
    EG = 1e3*float(r['E_Gamma'])
    q = EXP[:, 0]; qf = np.where(q <= G/2, q, G - q)
    Es = np.interp(qf, dk, E) - EG; Ese = np.interp(qf, dk, Ee)
    s2 = EXP[:, 2]**2 + Ese**2
    off = np.sum((EXP[:, 1] - Es)/s2)/np.sum(1/s2)          # fitted E_Gamma (meV)
    res = (EXP[:, 1] - Es - off)/np.sqrt(s2)
    chi2 = np.sum(res**2)
    if verbose:
        print(f'{prefix}: fitted E_Gamma = {off:.1f} meV, chi2 = {chi2:.2f} (dof 7), '
              f'model span along G-M = {E.max()-E.min():.0f} meV (min {E.min()-EG:+.0f}, max {E.max()-EG:+.0f} rel. E_Gamma)')
        for (qq, e, s), qq2, es, rr in zip(EXP, qf, Es + off, res):
            print(f'   dK={qq:.2f} (fold {qq2:.3f})  exp {e:6.1f}+-{s:4.1f}   model {es:6.1f}   resid {rr:+.2f} sigma')
    return off, chi2, dk, E - EG, Ee
if __name__ == '__main__':
    for p in sys.argv[1:]: compare(p)
