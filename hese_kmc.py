#!/usr/bin/env python3
r"""
hese_kmc.py
===========

Kinetic Monte Carlo (KMC) of an interacting lattice gas, and the coherent
intermediate scattering function (ISF) measured in helium spin-echo (HeSE).

For every temperature the script produces the dephasing-rate curve alpha(dK),
then performs an Arrhenius analysis at every dK to obtain the *effective*
activation energy E_eff(dK) = -d ln(alpha) / d(1/kT), and compares it with the
analytic first-cumulant (de Gennes) model

    alpha(dK,T) ~= Omega(dK,T) / S(dK,T)                       (1)
    Omega(dK,T) = sum_hops <rate> [1 - cos(dK.l)] / N          (CE form, exact)
    E_eff(dK)   = E_Gamma + d ln S(dK,beta) / d beta            (2)

with an analytic static structure factor of a lattice gas with pair potential
V(r) and lattice Fourier transform V(q) = sum_r V(r) cos(q.r):

    RPA:            S(q) = (1-theta) / [1 + beta theta(1-theta) V(q)]
    =>   E_eff(q) - E_Gamma = - theta(1-theta) V(q) / [1 + beta theta(1-theta) V(q)]
         (weak coupling: E_eff(q) - E_Gamma ~= - theta(1-theta) V(q))
    spherical RPA:  S(q) = 1 / {theta [lambda(T) + beta V(q)]}, with lambda(T) fixed by
                    the exact sum rule  <S_site(q)>_BZ = theta(1-theta)
                    (removes the spurious RPA divergence near ordering wave vectors)

Lattices
--------
Bravais : 'hexagonal', 'square', 'rectangular', 'centered_rectangular', 'oblique'
Non-Bravais : 'honeycomb'  (two inequivalent hollow sites, e.g. hcp + fcc on
              (111)/(0001) surfaces, with a site-energy difference dE).
Custom lattices: build a `Lattice` object (any basis, any hop rule).

Rates (detailed balance is satisfied exactly)
----------------------------------------------
A hop i->j (j empty, neighbour within `hop_cutoff`) has rate

    k = nu0 * exp(-beta * [E_TS0(b_i,b_j) - eps_{b_i} + lam*(E_i + E_j')/2 - E_i])

E_i  = interaction energy of the hopping particle at its initial site,
E_j' = interaction energy it would have at the final site (excluding itself).
lam = 1 : symmetric (BEP, gamma = 1/2) rule, barrier shift = (E_j' - E_i)/2
lam = 0 : initial-state rule, barrier shift = -E_i
Any lam gives detailed balance, because the transition-state energy is a
symmetric function of the initial and final environments.

Back-ends
---------
The KMC kernel exists twice with identical logic: as a small C library
(compiled on first use with the system C compiler and loaded with ctypes) and
as numba-compilable Python.  `--backend auto` tries C, then numba, then plain
Python (plain Python is ~100x slower; fine only for tests).

Units: energies eV, lengths Angstrom, time 1/nu0 (nu0 given in ps^-1 -> ps).

Usage examples
--------------
    python hese_kmc.py --preset hex_dipolar                 # main demonstration
    python hese_kmc.py --preset hex_dipolar --control       # same, no interactions
    python hese_kmc.py --preset honeycomb_ORu               # O/Ru(0001)-like, hcp+fcc
    python hese_kmc.py --preset square_dipolar --coverage 0.08 --temps 150 200 250
    python hese_kmc.py --preset hex_dipolar --workers 8 --prod-hops 50000
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

KB = 8.617333262e-5  # Boltzmann constant, eV/K


# =============================================================================
# 1. Lattices
# =============================================================================
@dataclass
class Lattice:
    """A 2D lattice with a basis.

    a1, a2       primitive vectors (Angstrom)
    basis        fractional coordinates of the basis sites, shape (nb, 2)
    site_energy  energy of an isolated adsorbate on each basis site (eV)
    hop_cutoff   hops connect sites closer than this distance (Angstrom)
    ts_energy    f(b_from, b_to, dvec) -> transition-state energy (eV), must be
                 symmetric under (b_from, b_to, dvec) -> (b_to, b_from, -dvec)
    n_rot, mirror  point-group operations used to average symmetry-equivalent
                 dK vectors (the ISF is real, so q and -q are always equivalent)
    azimuths     named scattering azimuths, Cartesian direction vectors
    """
    name: str
    a1: np.ndarray
    a2: np.ndarray
    basis: np.ndarray
    site_energy: np.ndarray
    sublattice_names: Tuple[str, ...]
    hop_cutoff: float
    ts_energy: Callable
    n_rot: int
    mirror: bool
    azimuths: Dict[str, np.ndarray]

    @property
    def nb(self) -> int:
        return len(self.basis)

    @property
    def bravais(self) -> bool:
        return self.nb == 1

    def basis_cart(self) -> np.ndarray:
        return np.asarray(self.basis, float) @ np.array([self.a1, self.a2])

    def recip(self) -> Tuple[np.ndarray, np.ndarray]:
        A = np.array([self.a1, self.a2]).T
        B = 2.0 * np.pi * np.linalg.inv(A).T
        return B[:, 0], B[:, 1]


def make_lattice(kind: str, a: float = 2.55, b: Optional[float] = None,
                 gamma_deg: float = 75.0, Ea: float = 0.10,
                 Ea2: Optional[float] = None, dE: float = 0.0,
                 hop_cutoff: Optional[float] = None) -> Lattice:
    """Preset lattices.

    hexagonal            fcc(111)/hcp(0001) top or single-hollow lattice, 6 neighbours
    square               fcc(001), 4 neighbours
    rectangular          e.g. fcc(110): a along x, b along y, barriers Ea (x), Ea2 (y)
    centered_rectangular e.g. bcc(110): a1=(a,0), a2=(a/2,b/2); hops to the 4
                         nearest neighbours (barrier Ea); set hop_cutoff to also
                         allow the hops along a1 (barrier Ea2)
    oblique              a1=(a,0), a2=b(cos g, sin g); hops to neighbours within hop_cutoff
    honeycomb            two hollow sites per cell (hcp at 1/3,1/3 and fcc at 2/3,2/3),
                         site energies (0, dE); hcp<->fcc hops with barrier Ea from hcp
                         (barrier Ea - dE from fcc)
    """
    Ea2 = Ea if Ea2 is None else Ea2
    s3 = math.sqrt(3.0)
    if kind == 'hexagonal':
        return Lattice('hexagonal', np.array([a, 0.0]), np.array([a / 2, a * s3 / 2]),
                       np.array([[0.0, 0.0]]), np.array([0.0]), ('site',),
                       hop_cutoff or 1.01 * a, lambda bf, bt, d: Ea, 6, True,
                       {'<1-10> (G-K)': np.array([1.0, 0.0]),
                        '<11-2> (G-M)': np.array([0.0, 1.0])})
    if kind == 'square':
        return Lattice('square', np.array([a, 0.0]), np.array([0.0, a]),
                       np.array([[0.0, 0.0]]), np.array([0.0]), ('site',),
                       hop_cutoff or 1.01 * a, lambda bf, bt, d: Ea, 4, True,
                       {'<110> (G-X)': np.array([1.0, 0.0]),
                        '<100> (G-M)': np.array([1.0, 1.0]) / math.sqrt(2)})
    if kind == 'rectangular':
        b = b if b is not None else a * math.sqrt(2)
        return Lattice('rectangular', np.array([a, 0.0]), np.array([0.0, b]),
                       np.array([[0.0, 0.0]]), np.array([0.0]), ('site',),
                       hop_cutoff or 1.01 * max(a, b),
                       lambda bf, bt, d: Ea if abs(d[1]) < 1e-6 else Ea2, 2, True,
                       {'G-X (along a1)': np.array([1.0, 0.0]),
                        'G-Y (along a2)': np.array([0.0, 1.0])})
    if kind == 'centered_rectangular':
        b = b if b is not None else a * math.sqrt(2)
        a1, a2 = np.array([a, 0.0]), np.array([a / 2, b / 2])
        dnn = min(a, np.linalg.norm(a2))
        return Lattice('centered_rectangular', a1, a2, np.array([[0.0, 0.0]]),
                       np.array([0.0]), ('site',), hop_cutoff or 1.01 * dnn,
                       lambda bf, bt, d: Ea if np.linalg.norm(d) < 1.01 * dnn else Ea2,
                       2, True, {'G-X (along a1)': np.array([1.0, 0.0]),
                                 'G-Y (along y)': np.array([0.0, 1.0])})
    if kind == 'oblique':
        b = b if b is not None else 1.2 * a
        g = math.radians(gamma_deg)
        a1, a2 = np.array([a, 0.0]), np.array([b * math.cos(g), b * math.sin(g)])
        return Lattice('oblique', a1, a2, np.array([[0.0, 0.0]]), np.array([0.0]),
                       ('site',), hop_cutoff or 1.01 * max(a, b), lambda bf, bt, d: Ea,
                       2, False, {'along a1': np.array([1.0, 0.0]),
                                  'perp. a1': np.array([0.0, 1.0])})
    if kind == 'honeycomb':
        if Ea <= dE:
            raise ValueError('honeycomb: barrier from the stable site must exceed dE')
        return Lattice('honeycomb', np.array([a, 0.0]), np.array([a / 2, a * s3 / 2]),
                       np.array([[1 / 3, 1 / 3], [2 / 3, 2 / 3]]), np.array([0.0, dE]),
                       ('hcp', 'fcc'), hop_cutoff or 1.01 * a / s3,
                       lambda bf, bt, d: Ea,      # TS energy measured from the hcp minimum
                       6, True, {'<11-20> (G-K)': np.array([1.0, 0.0]),
                                 '<1-100> (G-M)': np.array([0.0, 1.0])})
    raise ValueError(f'unknown lattice {kind}')


# =============================================================================
# 2. Pair potentials
# =============================================================================
class PairPotential:
    """V(b1, b2, r) in eV with cutoff r_cut (Angstrom)."""
    r_cut: float = 0.0

    def __call__(self, b1: int, b2: int, r: float) -> float:
        raise NotImplementedError


class NoPotential(PairPotential):
    r_cut = 0.0

    def __call__(self, b1, b2, r):
        return 0.0


class PowerPotential(PairPotential):
    """V(r) = V0 (r0/r)^p for r <= r_cut.  p = 3: dipole-dipole repulsion."""

    def __init__(self, V0: float, r0: float, p: float = 3.0, r_cut: float = 5.0):
        self.V0, self.r0, self.p, self.r_cut = V0, r0, p, r_cut

    def __call__(self, b1, b2, r):
        return self.V0 * (self.r0 / r) ** self.p if r <= self.r_cut + 1e-6 else 0.0


class ShellPotential(PairPotential):
    """Pair energies per neighbour shell, separately for each sublattice pair.

    shells: {(b1, b2): [V_1st, V_2nd, ...]} with b1 <= b2.  Shell distances are
    determined from the lattice (distinct distances between the two sublattices).
    """

    def __init__(self, lat: Lattice, shells: Dict[Tuple[int, int], List[float]]):
        self.table = {}
        rmax = 0.0
        bc = lat.basis_cart()
        for (b1, b2), vals in shells.items():
            b1, b2 = min(b1, b2), max(b1, b2)
            d = []
            for i in range(-6, 7):
                for j in range(-6, 7):
                    r = np.linalg.norm(i * lat.a1 + j * lat.a2 + bc[b2] - bc[b1])
                    if r > 1e-6:
                        d.append(round(r, 4))
            dist = sorted(set(d))[:len(vals)]
            self.table[(b1, b2)] = list(zip(dist, vals))
            rmax = max(rmax, dist[-1])
        self.r_cut = rmax + 1e-3

    def __call__(self, b1, b2, r):
        for d, v in self.table.get((min(b1, b2), max(b1, b2)), []):
            if abs(r - d) < 1e-3:
                return v
        return 0.0


def build_potential(spec: dict, lat: Lattice) -> PairPotential:
    a = float(np.linalg.norm(lat.a1))
    t = spec.get('type', 'none')
    if t == 'none':
        return NoPotential()
    if t in ('power', 'dipolar'):
        return PowerPotential(spec['V0'], spec.get('r0', 1.0) * a, spec.get('p', 3.0),
                              spec.get('r_cut', 5.0) * a)
    if t == 'shells':
        names = list(lat.sublattice_names)
        sh = {}
        for k, v in spec['shells'].items():
            n1, n2 = k.split('-')
            sh[(names.index(n1), names.index(n2))] = v
        return ShellPotential(lat, sh)
    raise ValueError(f'unknown potential type {t}')


# =============================================================================
# 3. Neighbour tables
# =============================================================================
def neighbour_stencil(lat: Lattice, R: float, include_self: bool = False):
    """For each basis site b: list of (di1, di2, b2, dvec) with |dvec| <= R."""
    bc = lat.basis_cart()
    area = abs(lat.a1[0] * lat.a2[1] - lat.a1[1] * lat.a2[0])
    n1 = int(math.ceil(R / (area / np.linalg.norm(lat.a2)))) + 1
    n2 = int(math.ceil(R / (area / np.linalg.norm(lat.a1)))) + 1
    out = []
    for b in range(lat.nb):
        lst = []
        for d1 in range(-n1, n1 + 1):
            for d2 in range(-n2, n2 + 1):
                for b2 in range(lat.nb):
                    d = d1 * lat.a1 + d2 * lat.a2 + bc[b2] - bc[b]
                    r = float(np.hypot(*d))
                    if r <= R + 1e-6 and (r > 1e-6 or include_self):
                        lst.append((d1, d2, b2, d))
        lst.sort(key=lambda x: (round(float(np.hypot(*x[3])), 6), x[2],
                                math.atan2(x[3][1], x[3][0])))
        out.append(lst)
    return out


def build_tables(lat: Lattice, L1: int, L2: int, pot: PairPotential) -> dict:
    nb = lat.nb
    n_sites = L1 * L2 * nb
    idx = np.arange(n_sites)
    b_of = idx % nb
    cell = idx // nb
    i1, i2 = cell // L2, cell % L2
    bc = lat.basis_cart()
    pos = i1[:, None] * lat.a1 + i2[:, None] * lat.a2 + bc[b_of]

    def sidx(j1, j2, b):
        return ((j1 % L1) * L2 + (j2 % L2)) * nb + b

    def fill(stencil, value_fn, include_value=True):
        nmax = max(len(s) for s in stencil)
        nbr = np.full((n_sites, max(nmax, 1)), -1, np.int32)
        val = np.zeros((n_sites, max(nmax, 1)))
        cnt = np.zeros(n_sites, np.int32)
        for b in range(nb):
            sb = idx[b_of == b]
            for m, (d1, d2, b2, d) in enumerate(stencil[b]):
                nbr[sb, m] = sidx(i1[sb] + d1, i2[sb] + d2, b2)
                if include_value:
                    val[sb, m] = value_fn(b, b2, d)
            cnt[sb] = len(stencil[b])
            # periodic images must not overlap
            t0 = nbr[sb[0], :cnt[sb[0]]]
            if len(set(t0.tolist())) != len(t0):
                raise ValueError('simulation cell too small for the stencil; increase L')
        return nbr, val, cnt

    # hops ---------------------------------------------------------------
    hop_st = neighbour_stencil(lat, lat.hop_cutoff)
    if min(len(s) for s in hop_st) == 0:
        raise ValueError('no hops found: check hop_cutoff')
    for b in range(nb):
        for (_, _, b2, d) in hop_st[b]:
            e1, e2 = lat.ts_energy(b, b2, d), lat.ts_energy(b2, b, -d)
            if abs(e1 - e2) > 1e-12:
                raise ValueError('ts_energy must be symmetric (detailed balance)')
    hop_nbr, hop_E0, n_hop = fill(hop_st, lambda b, b2, d: lat.ts_energy(b, b2, d) - lat.site_energy[b])
    _, hop_Vij, _ = fill(hop_st, lambda b, b2, d: pot(b, b2, float(np.hypot(*d))))
    zmax = hop_nbr.shape[1]
    hop_vec = np.zeros((nb, zmax, 2))
    hop_to = np.zeros((nb, zmax), np.int32)
    for b in range(nb):
        for h, (_, _, b2, d) in enumerate(hop_st[b]):
            hop_vec[b, h] = d
            hop_to[b, h] = b2

    # interactions -------------------------------------------------------
    if pot.r_cut > 0:
        st = neighbour_stencil(lat, pot.r_cut)
        st = [[x for x in s if abs(pot(b, x[2], float(np.hypot(*x[3])))) > 0] for b, s in enumerate(st)]
    else:
        st = [[] for _ in range(nb)]
    if max(len(s) for s in st) > 0:
        int_nbr, int_V, n_int = fill(st, lambda b, b2, d: pot(b, b2, float(np.hypot(*d))))
    else:
        int_nbr = np.full((n_sites, 1), -1, np.int32)
        int_V = np.zeros((n_sites, 1))
        n_int = np.zeros(n_sites, np.int32)
    int_stencil = st

    # update stencil (sites whose particles' rates can change after a hop) --
    rup = max(pot.r_cut, 0.0) + lat.hop_cutoff
    ust = neighbour_stencil(lat, rup, include_self=True)
    upd_nbr, _, n_upd = fill(ust, None, include_value=False)

    return dict(lat_name=lat.name, nb=nb, L1=L1, L2=L2, a1=lat.a1, a2=lat.a2,
                recip=np.array(lat.recip()), n_sites=n_sites, n_cells=L1 * L2,
                pos=pos, sub=b_of.astype(np.int32),
                hop_nbr=hop_nbr, hop_E0=np.ascontiguousarray(hop_E0),
                hop_Vij=np.ascontiguousarray(hop_Vij), n_hop=n_hop, zmax=zmax,
                hop_vec=hop_vec, hop_to=hop_to,
                int_nbr=int_nbr, int_V=np.ascontiguousarray(int_V), n_int=n_int,
                upd_nbr=upd_nbr, n_upd=n_upd,
                int_stencil=[[(x[2], np.array(x[3]), pot(b, x[2], float(np.hypot(*x[3]))))
                              for x in s] for b, s in enumerate(int_stencil)],
                site_energy=np.asarray(lat.site_energy, float))


# =============================================================================
# 4. Momentum transfers commensurate with the periodic cell
# =============================================================================
def build_q_set(lat: Lattice, L1: int, L2: int, dk_max: Optional[float] = None):
    """Allowed dK vectors along each azimuth plus their symmetry equivalents."""
    b1, b2 = lat.recip()
    a1, a2 = lat.a1, lat.a2
    shells, qvecs = [], []

    def commensurate(q):
        m1 = q @ a1 * L1 / (2 * np.pi)
        m2 = q @ a2 * L2 / (2 * np.pi)
        return abs(m1 - round(m1)) < 1e-6 and abs(m2 - round(m2)) < 1e-6

    ops = []
    for k in range(lat.n_rot):
        c, s = math.cos(2 * np.pi * k / lat.n_rot), math.sin(2 * np.pi * k / lat.n_rot)
        R = np.array([[c, -s], [s, c]])
        ops.append(R)
        if lat.mirror:
            ops.append(R @ np.diag([1.0, -1.0]))

    for name, u in lat.azimuths.items():
        u = np.asarray(u, float) / np.linalg.norm(u)
        # shortest reciprocal-lattice vector along u
        G = None
        for n1 in range(-12, 13):
            for n2 in range(-12, 13):
                g = n1 * b1 + n2 * b2
                ng = np.linalg.norm(g)
                if ng > 1e-9 and abs(u[0] * g[1] - u[1] * g[0]) < 1e-8 * ng and g @ u > 0:
                    if G is None or ng < G:
                        G = ng
        if G is None:
            G = 2 * np.pi / min(np.linalg.norm(a1), np.linalg.norm(a2)) * 2
        qmax = dk_max if dk_max is not None else (0.5 * G * 1.0001 if lat.bravais else G * 1.0001)
        M1 = int(math.ceil(qmax * np.linalg.norm(a1) * L1 / (2 * np.pi))) + 1
        M2 = int(math.ceil(qmax * np.linalg.norm(a2) * L2 / (2 * np.pi))) + 1
        cand = []
        for m1 in range(-M1, M1 + 1):
            for m2 in range(-M2, M2 + 1):
                q = m1 * b1 / L1 + m2 * b2 / L2
                nq = np.linalg.norm(q)
                if 1e-9 < nq <= qmax and abs(u[0] * q[1] - u[1] * q[0]) < 1e-8 * nq and q @ u > 0:
                    cand.append(q)
        cand.sort(key=lambda q: np.linalg.norm(q))
        for q in cand:
            eq = []
            for O in ops:
                qq = O @ q
                if not commensurate(qq):
                    continue
                if any(np.linalg.norm(qq - e) < 1e-8 or np.linalg.norm(qq + e) < 1e-8 for e in eq):
                    continue
                eq.append(qq)
            ids = list(range(len(qvecs), len(qvecs) + len(eq)))
            qvecs.extend(eq)
            shells.append(dict(az=name, dk=float(np.linalg.norm(q)), q=q, idx=ids))
    return shells, np.array(qvecs)


# =============================================================================
# 5. KMC kernels (numba-compilable Python and C, identical logic)
# =============================================================================
try:
    import numba
    from numba import njit
    HAVE_NUMBA = True
except Exception:  # pragma: no cover
    HAVE_NUMBA = False

    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return lambda f: f


@njit(cache=True)
def _py_seed(seed):
    np.random.seed(seed)


@njit(cache=True)
def _py_prates(p, part_site, occ, Eint, hop_nbr, n_hop, hop_E0, hop_Vij,
               beta, nu0, lam, rates, prate):
    s = part_site[p]
    Ei = Eint[s]
    tot = 0.0
    for h in range(n_hop[s]):
        t = hop_nbr[s, h]
        if occ[t] != 0:
            r = 0.0
        else:
            Ej = Eint[t] - hop_Vij[s, h]
            bar = hop_E0[s, h] + 0.5 * lam * (Ei + Ej) - Ei
            r = nu0 * math.exp(-beta * bar)
        rates[p, h] = r
        tot += r
    prate[p] = tot


@njit(cache=True)
def _py_all_rates(part_site, occ, Eint, hop_nbr, n_hop, hop_E0, hop_Vij,
                  beta, nu0, lam, rates, prate):
    for p in range(part_site.shape[0]):
        _py_prates(p, part_site, occ, Eint, hop_nbr, n_hop, hop_E0, hop_Vij,
                   beta, nu0, lam, rates, prate)


@njit(cache=True)
def _py_kmc(occ, site_part, part_site, sub, Eint, rates, prate,
            hop_nbr, n_hop, hop_E0, hop_Vij, int_nbr, int_V, n_int, upd_nbr, n_upd,
            phase, rho, buf, hop_count, stamp, beta, nu0, lam, times, dt_s, nbuf,
            t_stop, max_events, counters):
    Np = part_site.shape[0]
    nq = rho.shape[0]
    t = times[0]
    tn = times[1]
    k = 0
    nev = 0
    while True:
        if nbuf == 0 and nev >= max_events:
            break
        R = 0.0
        for p in range(Np):
            R += prate[p]
        if R <= 0.0:
            break
        dtev = -math.log(1.0 - np.random.random()) / R
        if nbuf > 0:
            full = False
            while tn < t + dtev:
                for iq in range(nq):
                    buf[k, iq] = rho[iq]
                k += 1
                if k == nbuf:
                    t = tn
                    tn += dt_s
                    full = True
                    break
                tn += dt_s
            if full:
                break
        else:
            if t + dtev >= t_stop:
                t = t_stop
                break
        t += dtev
        # --- choose event (BKL / n-fold way)
        x = np.random.random() * R
        acc = 0.0
        p = 0
        while p < Np:
            acc += prate[p]
            if acc > x:
                break
            p += 1
        if p == Np:
            p = Np - 1
            while prate[p] == 0.0:
                p -= 1
            xr = 0.5 * prate[p]
        else:
            xr = x - (acc - prate[p])
        s = part_site[p]
        nh = n_hop[s]
        a2 = 0.0
        h = 0
        while h < nh:
            a2 += rates[p, h]
            if a2 > xr:
                break
            h += 1
        if h == nh:
            h = nh - 1
            while rates[p, h] == 0.0:
                h -= 1
        tt = hop_nbr[s, h]
        # --- apply
        occ[s] = 0
        occ[tt] = 1
        site_part[s] = -1
        site_part[tt] = p
        part_site[p] = tt
        for m in range(n_int[s]):
            Eint[int_nbr[s, m]] -= int_V[s, m]
        for m in range(n_int[tt]):
            Eint[int_nbr[tt, m]] += int_V[tt, m]
        for iq in range(nq):
            rho[iq] += phase[tt, iq] - phase[s, iq]
        stamp_id = stamp[Np] + 1          # stamp[Np] holds the running event id
        stamp[Np] = stamp_id
        for m in range(n_upd[s]):
            q = site_part[upd_nbr[s, m]]
            if q >= 0 and stamp[q] != stamp_id:
                stamp[q] = stamp_id
                _py_prates(q, part_site, occ, Eint, hop_nbr, n_hop, hop_E0, hop_Vij,
                           beta, nu0, lam, rates, prate)
        for m in range(n_upd[tt]):
            q = site_part[upd_nbr[tt, m]]
            if q >= 0 and stamp[q] != stamp_id:
                stamp[q] = stamp_id
                _py_prates(q, part_site, occ, Eint, hop_nbr, n_hop, hop_E0, hop_Vij,
                           beta, nu0, lam, rates, prate)
        hop_count[sub[s], h] += 1
        nev += 1
    times[0] = t
    times[1] = tn
    counters[0] = k
    counters[1] = nev


C_SOURCE = r"""
#include <stdint.h>
#include <math.h>

static uint64_t rs[4];
static inline uint64_t rotl(const uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
static uint64_t splitmix64(uint64_t *x) {
    uint64_t z = (*x += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}
void rng_seed(uint64_t seed) { uint64_t x = seed; for (int i = 0; i < 4; i++) rs[i] = splitmix64(&x); }
static inline double rnd(void) {           /* xoshiro256+, uniform in [0,1) */
    const uint64_t result = rs[0] + rs[3];
    const uint64_t t = rs[1] << 17;
    rs[2] ^= rs[0]; rs[3] ^= rs[1]; rs[1] ^= rs[2]; rs[0] ^= rs[3];
    rs[2] ^= t; rs[3] = rotl(rs[3], 45);
    return (double)(result >> 11) * 0x1.0p-53;
}

static void prates(int64_t p, int64_t zmax, const int32_t *part_site, const int8_t *occ,
                   const double *Eint, const int32_t *hop_nbr, const int32_t *n_hop,
                   const double *hop_E0, const double *hop_Vij,
                   double beta, double nu0, double lam, double *rates, double *prate) {
    int64_t s = part_site[p];
    double Ei = Eint[s], tot = 0.0;
    for (int64_t h = 0; h < n_hop[s]; h++) {
        int64_t t = hop_nbr[s * zmax + h];
        double r;
        if (occ[t]) r = 0.0;
        else {
            double Ej = Eint[t] - hop_Vij[s * zmax + h];
            double bar = hop_E0[s * zmax + h] + 0.5 * lam * (Ei + Ej) - Ei;
            r = nu0 * exp(-beta * bar);
        }
        rates[p * zmax + h] = r;
        tot += r;
    }
    prate[p] = tot;
}

void all_rates(int64_t Np, int64_t zmax, const int32_t *part_site, const int8_t *occ,
               const double *Eint, const int32_t *hop_nbr, const int32_t *n_hop,
               const double *hop_E0, const double *hop_Vij,
               double beta, double nu0, double lam, double *rates, double *prate) {
    for (int64_t p = 0; p < Np; p++)
        prates(p, zmax, part_site, occ, Eint, hop_nbr, n_hop, hop_E0, hop_Vij,
               beta, nu0, lam, rates, prate);
}

void kmc_run(int64_t Np, int64_t zmax, int64_t imax, int64_t umax, int64_t nq,
             int8_t *occ, int32_t *site_part, int32_t *part_site, const int32_t *sub,
             double *Eint, double *rates, double *prate,
             const int32_t *hop_nbr, const int32_t *n_hop, const double *hop_E0,
             const double *hop_Vij, const int32_t *int_nbr, const double *int_V,
             const int32_t *n_int, const int32_t *upd_nbr, const int32_t *n_upd,
             const double *phase, double *rho, double *buf, int64_t *hop_count,
             int64_t *stamp, double beta, double nu0, double lam, double *times, double dt_s,
             int64_t nbuf, double t_stop, int64_t max_events, int64_t *counters) {
    double t = times[0], tn = times[1];
    int64_t k = 0, nev = 0;
    for (;;) {
        if (nbuf == 0 && nev >= max_events) break;
        double R = 0.0;
        for (int64_t p = 0; p < Np; p++) R += prate[p];
        if (R <= 0.0) break;
        double dtev = -log(1.0 - rnd()) / R;
        if (nbuf > 0) {
            int full = 0;
            while (tn < t + dtev) {
                double *dst = buf + 2 * k * nq;
                for (int64_t i = 0; i < 2 * nq; i++) dst[i] = rho[i];
                k++;
                if (k == nbuf) { t = tn; tn += dt_s; full = 1; break; }
                tn += dt_s;
            }
            if (full) break;
        } else if (t + dtev >= t_stop) { t = t_stop; break; }
        t += dtev;
        /* choose event */
        double x = rnd() * R, acc = 0.0, xr;
        int64_t p;
        for (p = 0; p < Np; p++) { acc += prate[p]; if (acc > x) break; }
        if (p == Np) { p = Np - 1; while (prate[p] == 0.0) p--; xr = 0.5 * prate[p]; }
        else xr = x - (acc - prate[p]);
        int64_t s = part_site[p], nh = n_hop[s], h;
        double a2 = 0.0;
        for (h = 0; h < nh; h++) { a2 += rates[p * zmax + h]; if (a2 > xr) break; }
        if (h == nh) { h = nh - 1; while (rates[p * zmax + h] == 0.0) h--; }
        int64_t tt = hop_nbr[s * zmax + h];
        /* apply */
        occ[s] = 0; occ[tt] = 1;
        site_part[s] = -1; site_part[tt] = (int32_t)p; part_site[p] = (int32_t)tt;
        for (int64_t m = 0; m < n_int[s]; m++) Eint[int_nbr[s * imax + m]] -= int_V[s * imax + m];
        for (int64_t m = 0; m < n_int[tt]; m++) Eint[int_nbr[tt * imax + m]] += int_V[tt * imax + m];
        for (int64_t i = 0; i < 2 * nq; i++) rho[i] += phase[2 * tt * nq + i] - phase[2 * s * nq + i];
        int64_t sid = ++stamp[Np];      /* stamp[Np] holds the running event id */
        for (int64_t m = 0; m < n_upd[s]; m++) {
            int64_t q = site_part[upd_nbr[s * umax + m]];
            if (q >= 0 && stamp[q] != sid) {
                stamp[q] = sid;
                prates(q, zmax, part_site, occ, Eint, hop_nbr, n_hop, hop_E0, hop_Vij,
                       beta, nu0, lam, rates, prate);
            }
        }
        for (int64_t m = 0; m < n_upd[tt]; m++) {
            int64_t q = site_part[upd_nbr[tt * umax + m]];
            if (q >= 0 && stamp[q] != sid) {
                stamp[q] = sid;
                prates(q, zmax, part_site, occ, Eint, hop_nbr, n_hop, hop_E0, hop_Vij,
                       beta, nu0, lam, rates, prate);
            }
        }
        hop_count[(int64_t)sub[s] * zmax + h] += 1;
        nev++;
    }
    times[0] = t; times[1] = tn; counters[0] = k; counters[1] = nev;
}
"""


class Backend:
    name = 'base'

    def seed(self, s): ...

    def all_rates(self, st, beta, nu0, lam): ...

    def run(self, st, buf, beta, nu0, lam, times, dt_s, nbuf, t_stop, max_events, counters): ...


class PyBackend(Backend):
    """numba (if installed) or plain-Python execution of the _py_* kernels."""

    def __init__(self, use_numba: bool):
        self.name = 'numba' if (use_numba and HAVE_NUMBA) else 'python'

    def seed(self, s):
        _py_seed(int(s) % (2 ** 32))

    def all_rates(self, st, beta, nu0, lam):
        T = st['tab']
        _py_all_rates(st['part_site'], st['occ'], st['Eint'], T['hop_nbr'], T['n_hop'],
                      T['hop_E0'], T['hop_Vij'], beta, nu0, lam, st['rates'], st['prate'])

    def run(self, st, buf, beta, nu0, lam, times, dt_s, nbuf, t_stop, max_events, counters):
        T = st['tab']
        _py_kmc(st['occ'], st['site_part'], st['part_site'], T['sub'], st['Eint'], st['rates'],
                st['prate'], T['hop_nbr'], T['n_hop'], T['hop_E0'], T['hop_Vij'], T['int_nbr'],
                T['int_V'], T['n_int'], T['upd_nbr'], T['n_upd'], T['phase'], st['rho'], buf,
                st['hop_count'], st['stamp'], beta, nu0, lam, times, dt_s, nbuf, t_stop,
                max_events, counters)


class CBackend(Backend):
    name = 'C'

    def __init__(self):
        h = hashlib.md5(C_SOURCE.encode()).hexdigest()[:12]
        d = os.path.join(tempfile.gettempdir(), 'hese_kmc_build')
        os.makedirs(d, exist_ok=True)
        so = os.path.join(d, f'kmc_{h}.so')
        if not os.path.exists(so):
            src = os.path.join(d, f'kmc_{h}.c')
            with open(src, 'w') as f:
                f.write(C_SOURCE)
            cc = os.environ.get('CC', 'cc')
            tmp = so + f'.{os.getpid()}'
            subprocess.run([cc, '-O3', '-fPIC', '-shared', src, '-o', tmp, '-lm'],
                           check=True, capture_output=True)
            os.replace(tmp, so)
        lib = ctypes.CDLL(so)
        P, I, D = ctypes.c_void_p, ctypes.c_int64, ctypes.c_double
        lib.rng_seed.argtypes = [ctypes.c_uint64]
        lib.all_rates.argtypes = [I, I] + [P] * 7 + [D, D, D, P, P]
        lib.kmc_run.argtypes = [I] * 5 + [P] * 21 + [D, D, D, P, D, I, D, I, P]
        self.lib = lib

    @staticmethod
    def _p(a):
        return a.ctypes.data_as(ctypes.c_void_p)

    def seed(self, s):
        self.lib.rng_seed(int(s) & 0xFFFFFFFFFFFFFFFF)

    def all_rates(self, st, beta, nu0, lam):
        T, p = st['tab'], self._p
        self.lib.all_rates(len(st['part_site']), T['zmax'], p(st['part_site']), p(st['occ']),
                           p(st['Eint']), p(T['hop_nbr']), p(T['n_hop']), p(T['hop_E0']),
                           p(T['hop_Vij']), beta, nu0, lam, p(st['rates']), p(st['prate']))

    def run(self, st, buf, beta, nu0, lam, times, dt_s, nbuf, t_stop, max_events, counters):
        T, p = st['tab'], self._p
        self.lib.kmc_run(len(st['part_site']), T['zmax'], T['int_nbr'].shape[1],
                         T['upd_nbr'].shape[1], len(st['rho']),
                         p(st['occ']), p(st['site_part']), p(st['part_site']), p(T['sub']),
                         p(st['Eint']), p(st['rates']), p(st['prate']),
                         p(T['hop_nbr']), p(T['n_hop']), p(T['hop_E0']), p(T['hop_Vij']),
                         p(T['int_nbr']), p(T['int_V']), p(T['n_int']), p(T['upd_nbr']),
                         p(T['n_upd']), p(T['phase']), p(st['rho']), p(buf),
                         p(st['hop_count']), p(st['stamp']), beta, nu0, lam, p(times), dt_s,
                         int(nbuf),
                         float(t_stop), int(max_events), p(counters))


def get_backend(name: str = 'auto') -> Backend:
    if name in ('auto', 'c'):
        try:
            return CBackend()
        except Exception as e:
            if name == 'c':
                raise
            print(f'[backend] C compilation unavailable ({type(e).__name__}); trying numba')
    if name in ('auto', 'numba') and HAVE_NUMBA:
        return PyBackend(True)
    if name == 'numba':
        raise RuntimeError('numba not installed')
    print('[backend] WARNING: pure-Python kernel (very slow). Install numba or a C compiler.')
    return PyBackend(False)


# =============================================================================
# 6. Replica state and simulation of one (T, seed)
# =============================================================================
def new_state(tab: dict, Np: int, rng: np.random.Generator) -> dict:
    n = tab['n_sites']
    sites = rng.choice(n, Np, replace=False).astype(np.int32)
    st = dict(tab=tab, occ=np.zeros(n, np.int8), site_part=np.full(n, -1, np.int32),
              part_site=sites.copy(), Eint=np.zeros(n),
              rates=np.zeros((Np, tab['zmax'])), prate=np.zeros(Np),
              rho=np.zeros(tab['phase'].shape[1], np.complex128),
              hop_count=np.zeros((tab['nb'], tab['zmax']), np.int64),
              stamp=np.zeros(Np + 1, np.int64))
    st['occ'][sites] = 1
    st['site_part'][sites] = np.arange(Np, dtype=np.int32)
    refresh_state(st)
    return st


def refresh_state(st: dict):
    """Recompute interaction field and density modes from scratch (kills round-off drift)."""
    tab = st['tab']
    s = st['part_site']
    st['Eint'][:] = 0.0
    ni = tab['n_int'][s]
    if ni.max() > 0:
        mask = np.arange(tab['int_nbr'].shape[1])[None, :] < ni[:, None]
        np.add.at(st['Eint'], tab['int_nbr'][s][mask], tab['int_V'][s][mask])
    st['rho'][:] = tab['phase'][s].sum(axis=0)


class Correlator:
    """Accumulates sum_s Re[X(s+tau) X*(s)] for tau < nlag over a stream of samples (FFT)."""

    def __init__(self, nlag: int, nq: int):
        self.nlag = nlag
        self.tail = np.zeros((0, nq), np.complex128)
        self.sum = np.zeros((nlag, nq))
        self.count = 0

    def add(self, X: np.ndarray):
        Y = np.concatenate([self.tail, X]) if len(self.tail) else X
        n_orig = len(Y) - (self.nlag - 1)
        if n_orig <= 0:
            self.tail = Y.copy()
            return
        nfft = 1 << int(math.ceil(math.log2(len(Y) + self.nlag)))
        F = np.fft.fft(Y, n=nfft, axis=0)
        F0 = np.fft.fft(Y[:n_orig], n=nfft, axis=0)
        self.sum += np.fft.ifft(F * np.conj(F0), axis=0)[:self.nlag].real
        self.count += n_orig
        self.tail = Y[n_orig:].copy()


_G: dict = {}


def _init_worker(tab, backend_name):
    _G['tab'] = tab
    _G['be'] = get_backend(backend_name)


def simulate_replica(job: dict) -> dict:
    """Equilibrate, then sample rho_q(t) on a uniform grid and correlate."""
    tab, be = _G['tab'], _G['be']
    T, seed, Np = job['T'], job['seed'], job['Np']
    beta, nu0, lam = 1.0 / (KB * T), job['nu0'], job['lam']
    rng = np.random.default_rng(seed)
    be.seed(seed * 7919 + 17)
    st = new_state(tab, Np, rng)
    be.all_rates(st, beta, nu0, lam)
    times = np.zeros(2)
    cnt = np.zeros(2, np.int64)
    dummy = np.zeros((1, len(st['rho'])), np.complex128)
    # equilibration (in pieces, refreshing the fields in between)
    n_piece = 10
    for _ in range(n_piece):
        be.run(st, dummy, beta, nu0, lam, times, 1.0, 0, times[0] + job['t_eq'] / n_piece,
               2 ** 62, cnt)
        refresh_state(st)
        be.all_rates(st, beta, nu0, lam)
    st['hop_count'][:] = 0
    times[1] = times[0]                      # first sample at the current time
    dt, chunk = job['dt'], job['chunk']
    cf = Correlator(job['nlag_f'], len(st['rho']))
    cc = Correlator(job['nlag_c'], len(st['rho']))
    buf = np.zeros((chunk, len(st['rho'])), np.complex128)
    remaining, gidx, n_events = job['n_samples'], 0, 0
    sub_occ = np.zeros(tab['nb'])
    n_occ = 0
    while remaining > 0:
        nb_ = min(chunk, remaining)
        be.run(st, buf, beta, nu0, lam, times, dt, nb_, 0.0, 0, cnt)
        k = int(cnt[0])
        n_events += int(cnt[1])
        X = buf[:k]
        cf.add(X)
        first = (-gidx) % job['n_sub']
        cc.add(X[first::job['n_sub']])
        gidx += k
        remaining -= k
        refresh_state(st)
        be.all_rates(st, beta, nu0, lam)
        sub_occ += np.bincount(tab['sub'][st['part_site']], minlength=tab['nb'])
        n_occ += 1
    return dict(T=T, seed=seed, Cf=cf.sum, nf=cf.count, Cc=cc.sum, nc=cc.count,
                hop_count=st['hop_count'].copy(), T_prod=gidx * dt, n_events=n_events,
                Np=Np, sub_frac=sub_occ / (n_occ * Np))


def pilot_rate(tab, be, Np, T, nu0, lam, seed=12345, hops=300) -> float:
    """Mean hop rate per particle, Gamma_eff(T), from a short run (sets the time grid)."""
    beta = 1.0 / (KB * T)
    st = new_state(tab, Np, np.random.default_rng(seed))
    be.seed(seed)
    be.all_rates(st, beta, nu0, lam)
    times, cnt = np.zeros(2), np.zeros(2, np.int64)
    dummy = np.zeros((1, len(st['rho'])), np.complex128)
    be.run(st, dummy, beta, nu0, lam, times, 1.0, 0, 1e300, hops * Np, cnt)   # equilibrate
    t0 = times[0]
    be.run(st, dummy, beta, nu0, lam, times, 1.0, 0, 1e300, hops * Np, cnt)
    return cnt[1] / (Np * (times[0] - t0))


# =============================================================================
# 7. Analysis: fits, Arrhenius, analytic models
# =============================================================================
def fit_decay(t, y, floor=math.exp(-2), model='single', skip=0, min_pts=6):
    """Fit A exp(-alpha t) (or a two-exponential) to the normalised ISF, using the
    window from t[skip] until the ISF first drops below `floor`.  Returns (alpha, A)
    or None if the window is unsuitable (too short / never decays)."""
    if y[0] <= 0:
        return None
    yn = y / y[0]
    below = np.nonzero(yn[skip:] < floor)[0]
    if len(below) == 0:
        return None
    iend = skip + below[0]
    if iend - skip < min_pts:
        return None
    tt, yy = t[skip:iend + 1], yn[skip:iend + 1]
    m = yy > 0
    pp = np.polyfit(tt[m], np.log(yy[m]), 1)
    a0, A0 = max(-pp[0], 1e-15), math.exp(pp[1])
    try:
        from scipy.optimize import curve_fit
        if model == 'single':
            popt, _ = curve_fit(lambda x, A, a: A * np.exp(-a * x), tt, yy, p0=[A0, a0], maxfev=5000)
            return float(popt[1]), float(popt[0])
        # double exponential over a longer window: report the slow (dominant) rate
        below2 = np.nonzero(yn[skip:] < floor ** 2)[0]
        i2 = skip + (below2[0] if len(below2) else len(yn) - 1 - skip)
        tt, yy = t[skip:i2 + 1], yn[skip:i2 + 1]
        f2 = lambda x, A1, a1, A2, a2: A1 * np.exp(-a1 * x) + A2 * np.exp(-a2 * x)
        popt, _ = curve_fit(f2, tt, yy, p0=[0.8 * A0, 0.8 * a0, 0.2, 5 * a0], maxfev=20000)
        A1, a1, A2, a2 = popt
        return (float(a1), float(A1)) if a1 < a2 else (float(a2), float(A2))
    except Exception:
        return a0, A0


def arrhenius(temps, alpha):
    x = 1.0 / (KB * np.asarray(temps))
    y = np.log(alpha)
    ok = np.isfinite(y)
    if ok.sum() < 2:
        return np.nan, np.nan
    sl, ic = np.polyfit(x[ok], y[ok], 1)
    return -sl, ic


def dilute_ce_modes(q, tab, nu0, T):
    """Generalised Chudley-Elliott (single particle, dilute, no interactions) for a
    lattice with a basis: returns decay rates and amplitudes of the ISF modes."""
    nb, beta = tab['nb'], 1.0 / (KB * T)
    p_eq = np.exp(-beta * tab['site_energy'])
    p_eq /= p_eq.sum()
    M = np.zeros((nb, nb), complex)
    s0 = [np.nonzero(tab['sub'] == b)[0][0] for b in range(nb)]
    for b in range(nb):
        for h in range(tab['n_hop'][s0[b]]):
            k = nu0 * math.exp(-beta * tab['hop_E0'][s0[b], h])
            b2 = tab['hop_to'][b, h]
            M[b2, b] += k * np.exp(-1j * q @ tab['hop_vec'][b, h])
            M[b, b] -= k
    w, V = np.linalg.eig(M)
    c = np.linalg.solve(V, p_eq.astype(complex))
    amp = (V.sum(axis=0) * c).real
    order = np.argsort(-w.real)
    return -w.real[order], amp[order]


def analyse(results: List[dict], shells, qvecs, tab, cfg) -> dict:
    temps = sorted(set(r['T'] for r in results), reverse=True)
    seeds = sorted(set(r['seed'] % 1000 for r in results))
    nS = len(seeds)
    nT, nsh = len(temps), len(shells)
    theta_site = results[0]['Np'] / tab['n_sites']
    hop_vec = tab['hop_vec']
    floor, model = cfg['fit_floor'], cfg['fit_model']

    def aggregate(rs):
        Cf = sum(r['Cf'] for r in rs); nf = sum(r['nf'] for r in rs)
        Cc = sum(r['Cc'] for r in rs); nc = sum(r['nc'] for r in rs)
        hc = sum(r['hop_count'] for r in rs); NT = sum(r['Np'] * r['T_prod'] for r in rs)
        Np = rs[0]['Np']
        Ff = np.array([Cf[:, sh['idx']].mean(axis=1) for sh in shells]).T / (max(nf, 1) * Np)
        Fc = np.array([Cc[:, sh['idx']].mean(axis=1) for sh in shells]).T / (max(nc, 1) * Np)
        S = Ff[0]
        Gam = hc.sum() / NT
        Om = np.zeros(nsh)
        for j, sh in enumerate(shells):
            cq = 1.0 - np.cos(np.einsum('bhx,x->bh', hop_vec, sh['q']))
            Om[j] = (hc * cq).sum() / NT
        return Ff, Fc, S, Gam, Om

    def rates_from(Ff, Fc, dt, nsub):
        tf = np.arange(Ff.shape[0]) * dt
        tc = np.arange(Fc.shape[0]) * dt * nsub
        al = np.full(nsh, np.nan)
        for j in range(nsh):
            r = fit_decay(tf, Ff[:, j], floor, model, cfg['fit_skip'])
            if r is None:
                r = fit_decay(tc, Fc[:, j], floor, model, 0, 4)
            if r is not None:
                al[j] = r[0]
        return al

    out = dict(temps=np.array(temps), dk=np.array([s['dk'] for s in shells]),
               az=[s['az'] for s in shells], theta_site=theta_site)
    keys = ['alpha', 'S', 'Gamma', 'Omega', 'alpha1']
    full = {k: [] for k in keys}
    jack = {k: [[] for _ in range(nS)] for k in keys}
    ISF = []
    for T in temps:
        rs = [r for r in results if r['T'] == T]
        dt, nsub = cfg['grid'][T]['dt'], cfg['grid'][T]['n_sub']
        Ff, Fc, S, Gam, Om = aggregate(rs)
        ISF.append(dict(tf=np.arange(Ff.shape[0]) * dt, Ff=Ff,
                        tc=np.arange(Fc.shape[0]) * dt * nsub, Fc=Fc))
        al = rates_from(Ff, Fc, dt, nsub)
        for k, v in zip(keys, [al, S, Gam, Om, Om / S]):
            full[k].append(v)
        for js, sd in enumerate(seeds):
            rj = [r for r in rs if r['seed'] % 1000 != sd]
            if not rj:
                rj = rs
            Ff, Fc, S, Gam, Om = aggregate(rj)
            al = rates_from(Ff, Fc, dt, nsub)
            for k, v in zip(keys, [al, S, Gam, Om, Om / S]):
                jack[k][js].append(v)
    for k in keys:
        out[k] = np.array(full[k])
        J = np.array([np.array(jack[k][j]) for j in range(nS)])
        out[k + '_err'] = np.sqrt((nS - 1) / nS * ((J - J.mean(0)) ** 2).sum(0)) if nS > 1 else 0 * out[k]
        out[k + '_jack'] = J
    out['ISF'] = ISF

    # Arrhenius at every dK -------------------------------------------------
    def eff(A):  # A: (nT, nsh)
        return np.array([arrhenius(temps, A[:, j])[0] for j in range(nsh)])

    for k in ['alpha', 'alpha1']:
        out['E_' + k] = eff(out[k])
        Ej = np.array([eff(out[k + '_jack'][j]) for j in range(nS)])
        out['E_' + k + '_err'] = np.sqrt((nS - 1) / nS * ((Ej - Ej.mean(0)) ** 2).sum(0)) if nS > 1 else 0 * out['E_' + k]
    out['E_Gamma'] = arrhenius(temps, out['Gamma'])[0]
    GJ = np.array([arrhenius(temps, out['Gamma_jack'][j])[0] for j in range(nS)])
    out['E_Gamma_err'] = float(np.sqrt((nS - 1) / nS * ((GJ - GJ.mean()) ** 2).sum())) if nS > 1 else 0.0
    out['E_S'] = np.array([arrhenius(temps, 1.0 / out['S'][:, j])[0] for j in range(nsh)])  # = dlnS/dbeta

    # analytic models -----------------------------------------------------
    nu0 = cfg['nu0']
    if tab['nb'] == 1:
        th = theta_site
        c = th * (1 - th)
        st = tab['int_stencil'][0]
        rv = np.array([x[1] for x in st]) if st else np.zeros((0, 2))
        Vv = np.array([x[2] for x in st]) if st else np.zeros(0)
        Vq = np.array([(Vv * np.cos(rv @ sh['q'])).sum() for sh in shells])
        nbz = 96
        b1, b2 = tab['recip']
        gq = (np.arange(nbz)[:, None, None] * b1 + np.arange(nbz)[None, :, None] * b2).reshape(-1, 2) / nbz
        Vbz = np.cos(gq @ rv.T) @ Vv if len(Vv) else np.zeros(len(gq))
        S_rpa, S_srpa, lams = [], [], []
        for T in temps:
            b = 1.0 / (KB * T)
            den = 1 + b * c * Vq
            S_rpa.append(np.where(den > 0.05, (1 - th) / np.maximum(den, 0.05), np.nan))
            # spherical RPA: solve <1/(lam + b V(q))>_BZ = theta(1-theta) by bisection
            lo = -b * Vbz.min() + 1e-12
            hi = lo + 1.0 / c + 1.0
            while np.mean(1.0 / (hi + b * Vbz)) > c:
                hi = lo + 2 * (hi - lo)
            for _ in range(200):
                mid = 0.5 * (lo + hi)
                if np.mean(1.0 / (mid + b * Vbz)) > c:
                    lo = mid
                else:
                    hi = mid
            lam_T = 0.5 * (lo + hi)
            lams.append(lam_T)
            S_srpa.append(1.0 / (lam_T + b * Vq) / th)
        S_rpa, S_srpa = np.array(S_rpa), np.array(S_srpa)
        out.update(Vq=Vq, S_rpa=S_rpa, S_srpa=S_srpa, lambda_srpa=np.array(lams),
                   alpha_rpa=out['Omega'] / S_rpa, alpha_srpa=out['Omega'] / S_srpa,
                   alpha_ce=out['Omega'] / (1 - th))
        out['E_rpa'] = eff(out['alpha_rpa'])
        out['E_srpa'] = eff(out['alpha_srpa'])
        out['E_lin'] = out['E_Gamma'] - c * Vq          # weak-coupling limit
    else:
        ce = np.array([[dilute_ce_modes(sh['q'], tab, nu0, T)[0][0] for sh in shells] for T in temps])
        out['alpha_ce'] = ce
        out['E_ce'] = eff(ce)
    out['sub_frac'] = np.array([np.mean([r['sub_frac'] for r in results if r['T'] == T], axis=0)
                                for T in temps])
    return out


# =============================================================================
# 8. Plotting
# =============================================================================
def make_plots(res: dict, cfg: dict, prefix: str, control: Optional[dict] = None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import cm

    temps, dk = res['temps'], res['dk']
    azs = list(dict.fromkeys(res['az']))
    cols = cm.plasma(np.linspace(0, 0.85, len(temps)))
    bravais = 'E_srpa' in res
    figs = []

    # Figure 1: dephasing-rate curves, absolute and normalised by Gamma_eff(T)
    fig, axs = plt.subplots(2, len(azs), figsize=(5.2 * len(azs), 8.2), squeeze=False)
    for ia, az in enumerate(azs):
        m = np.array([a == az for a in res['az']])
        ax = axs[0, ia]
        for it, T in enumerate(temps):
            ax.errorbar(dk[m], res['alpha'][it, m], res['alpha_err'][it, m], fmt='o-', ms=3.5,
                        color=cols[it], label=f'{T:.0f} K')
            ax.plot(dk[m], res['alpha_ce'][it, m], ':', color=cols[it], lw=1)
        ax.set_yscale('log')
        ax.set_title(f'{az}: dephasing rate (dotted: CE, same jump rate)')
        ax.set_xlabel(r'$\Delta K$ ($\AA^{-1}$)')
        ax.set_ylabel(r'$\alpha$ (ps$^{-1}$)')
        if ia == 0:
            ax.legend(fontsize=8, ncol=2)
        ax = axs[1, ia]
        for it, T in enumerate(temps):
            ax.plot(dk[m], res['alpha'][it, m] / res['Gamma'][it], 'o-', ms=3.5, color=cols[it],
                    label=f'{T:.0f} K')
        if bravais:
            ax.plot(dk[m], res['alpha_ce'][0, m] / res['Gamma'][0], 'k--', lw=1.5,
                    label='CE (no interactions)')
        ax.set_title(r'shape: $\alpha(\Delta K,T)\,/\,\Gamma_{\rm eff}(T)$')
        ax.set_xlabel(r'$\Delta K$ ($\AA^{-1}$)')
        ax.set_ylabel(r'$\alpha/\Gamma_{\rm eff}$')
        ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(prefix + '_dephasing_curves.png', dpi=140)
    figs.append(prefix + '_dephasing_curves.png')
    plt.close(fig)

    # Figure 2: Arrhenius plots at selected dK
    fig, axs = plt.subplots(1, len(azs), figsize=(5.2 * len(azs), 4.4), squeeze=False)
    x = 1000.0 / temps
    for ia, az in enumerate(azs):
        ax = axs[0, ia]
        js = np.nonzero([a == az for a in res['az']])[0]
        pick = js[np.unique(np.linspace(0, len(js) - 1, 5).round().astype(int))]
        cc = cm.viridis(np.linspace(0, 0.9, len(pick)))
        for c, j in zip(cc, pick):
            y = res['alpha'][:, j]
            ax.errorbar(x, y, res['alpha_err'][:, j], fmt='o', color=c, ms=5)
            E, ic = arrhenius(temps, y)
            xx = np.linspace(x.min(), x.max(), 50)
            ax.plot(xx, np.exp(ic - E / (KB * 1000.0 / xx)), '-', color=c,
                    label=fr'$\Delta K$={dk[j]:.2f}: $E_{{\rm eff}}$={1e3 * E:.1f}$\pm${1e3 * res["E_alpha_err"][j]:.1f} meV')
        ax.set_yscale('log')
        ax.set_xlabel('1000/T (K$^{-1}$)')
        ax.set_ylabel(r'$\alpha$ (ps$^{-1}$)')
        ax.set_title(f'Arrhenius plots, {az}')
        ax.legend(fontsize=7.5)
    fig.tight_layout()
    fig.savefig(prefix + '_arrhenius.png', dpi=140)
    figs.append(prefix + '_arrhenius.png')
    plt.close(fig)

    # Figure 3: E_eff(dK) and S(dK,T)
    fig, axs = plt.subplots(2, len(azs), figsize=(5.2 * len(azs), 8.2), squeeze=False)
    for ia, az in enumerate(azs):
        m = np.array([a == az for a in res['az']])
        ax = axs[0, ia]
        ax.errorbar(dk[m], 1e3 * res['E_alpha'][m], 1e3 * res['E_alpha_err'][m], fmt='o', color='k',
                    ms=4.5, label='KMC: fitted $\\alpha$', zorder=5)
        ax.plot(dk[m], 1e3 * res['E_alpha1'][m], 's', mfc='none', color='C3', ms=5,
                label=r'KMC: $\Omega/S$ (first cumulant)')
        ax.axhline(1e3 * float(res['E_Gamma']), color='0.5', ls='--', lw=1,
                   label=r'$E_\Gamma$ (mean jump rate)')
        if bravais:
            ax.plot(dk[m], 1e3 * res['E_srpa'][m], '-', color='C0', lw=2, label='analytic: spherical RPA')
            ax.plot(dk[m], 1e3 * res['E_rpa'][m], ':', color='C2', lw=2, label='analytic: RPA')
        else:
            ax.plot(dk[m], 1e3 * res['E_ce'][m], '-', color='C0', lw=1.5,
                    label='dilute CE (slow mode), no interactions')
        if control is not None:
            ax.errorbar(control['dk'][m], 1e3 * control['E_alpha'][m], 1e3 * control['E_alpha_err'][m],
                        fmt='^', color='C1', ms=4, label='KMC without interactions')
        ax.set_xlabel(r'$\Delta K$ ($\AA^{-1}$)')
        ax.set_ylabel(r'$E_{\rm eff}$ (meV)')
        ax.set_title(f'effective activation energy, {az}')
        ax.legend(fontsize=7.5)
        ax = axs[1, ia]
        for it, T in enumerate(temps):
            ax.plot(dk[m], res['S'][it, m], 'o', ms=3.5, color=cols[it], label=f'{T:.0f} K')
            if bravais:
                ax.plot(dk[m], res['S_srpa'][it, m], '-', color=cols[it], lw=1)
        ax.set_xlabel(r'$\Delta K$ ($\AA^{-1}$)')
        ax.set_ylabel(r'$S(\Delta K)$ per adsorbate')
        ax.set_title('static structure factor (points: KMC, lines: spherical RPA)' if bravais
                     else 'static structure factor (KMC)')
        ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(prefix + '_Eeff.png', dpi=140)
    figs.append(prefix + '_Eeff.png')
    plt.close(fig)
    return figs


# =============================================================================
# 9. Presets and driver
# =============================================================================
PRESETS = {
    # Polar adsorbate on a (111) surface: dipole-dipole repulsion V(r) = V0 (a/r)^3,
    # V0 = 25 meV at a = 2.55 A corresponds to an effective dipole of ~0.8 D.
    'hex_dipolar': dict(lattice='hexagonal', a=2.55, Ea=0.100, nu0=1.0, lam=1.0, coverage=0.15,
                        potential=dict(type='dipolar', V0=0.025, p=3, r_cut=5.0),
                        temps=[300, 244, 206, 178, 157, 140], L=(48, 48)),
    # Stronger, alkali-like dipoles at lower coverage (beyond the range of validity of RPA)
    'hex_dipolar_strong': dict(lattice='hexagonal', a=2.55, Ea=0.100, nu0=1.0, lam=1.0,
                               coverage=0.10, potential=dict(type='dipolar', V0=0.100, p=3, r_cut=5.0),
                               temps=[300, 244, 206, 178, 157, 140], L=(48, 48)),
    'square_dipolar': dict(lattice='square', a=2.55, Ea=0.100, nu0=1.0, lam=1.0, coverage=0.10,
                           potential=dict(type='dipolar', V0=0.050, p=3, r_cut=5.0),
                           temps=[300, 244, 206, 178, 157, 140], L=(40, 40)),
    'rect_dipolar': dict(lattice='rectangular', a=2.55, b=3.61, Ea=0.080, Ea2=0.120, nu0=1.0,
                         lam=1.0, coverage=0.10, potential=dict(type='dipolar', V0=0.05, p=3, r_cut=4.0),
                         temps=[300, 244, 206, 178, 157, 140], L=(40, 30)),
    # O/Ru(0001), hcp sites only (fcc ~0.45 eV higher, ignored). Pair energies by shell
    # (a, sqrt3 a, 2a, sqrt7 a): V(a) from Stampfl et al. (DFT), the rest from the STM dimer
    # potential of Renisch et al., PRL 82, 3839 (1999). Ea only offsets E_eff(dK).
    'hex_ORu': dict(lattice='hexagonal', a=2.71, Ea=0.50, nu0=10.0, lam=1.0, coverage=0.06,
                    potential=dict(type='shells', shells={'site-site': [0.265, 0.030, -0.048, -0.025]}),
                    temps=[781, 806, 833, 862, 893, 926], L=(48, 48)),
    # O/Ru(0001)-like honeycomb: hcp + fcc hollows. Site energy and pair interactions per
    # sublattice pair from the DFT lattice-gas parameters of McEwen, Payne & Stampfl,
    # Chem. Phys. Lett. 361, 317 (2002), trio terms omitted.  The hcp->fcc barrier is an
    # illustrative value (adjust to your system).
    'honeycomb_ORu': dict(lattice='honeycomb', a=2.71, Ea=0.68, dE=0.351, nu0=10.0, lam=1.0,
                          coverage=0.10,
                          potential=dict(type='shells', shells={'hcp-hcp': [0.265, 0.044, -0.025],
                                                                'fcc-fcc': [0.158, 0.016, 0.002],
                                                                'hcp-fcc': [0.586, 0.101, 0.033]}),
                          temps=[1000, 920, 850, 790, 740, 700], L=(36, 36)),
}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--preset', default='hex_dipolar', choices=list(PRESETS))
    ap.add_argument('--lattice', help='override lattice kind')
    ap.add_argument('--coverage', type=float, help='adsorbates per primitive cell (ML)')
    ap.add_argument('--temps', type=float, nargs='+')
    ap.add_argument('--L', type=int, nargs='+', help='cell repeats (L or L1 L2)')
    ap.add_argument('--V0', type=float, help='power-law/dipolar strength at r0 (eV)')
    ap.add_argument('--potential-json', help='replace the potential spec, e.g. '
                    '\'{"type":"shells","shells":{"site-site":[0.265,0.03,-0.048,-0.025]}}\'')
    ap.add_argument('--Ea', type=float, help='barrier (eV); TS energy above the most stable site')
    ap.add_argument('--dE', type=float, help='honeycomb site-energy difference (eV)')
    ap.add_argument('--lam', type=float, help='TS coupling: 1 = symmetric BEP, 0 = initial-state rule')
    ap.add_argument('--control', action='store_true', help='switch interactions off')
    ap.add_argument('--n-seeds', type=int, default=8, help='independent replicas per T (jackknife)')
    ap.add_argument('--prod-hops', type=float, default=2.0e4,
                    help='production length per replica, in hops per particle')
    ap.add_argument('--eq-hops', type=float, default=500.0)
    ap.add_argument('--dt-factor', type=float, default=0.05, help='sampling step x Gamma_eff')
    ap.add_argument('--nlag', type=int, default=512, help='fine-grid lags')
    ap.add_argument('--n-sub', type=int, default=16, help='coarse grid = n_sub x fine grid')
    ap.add_argument('--fit-floor', type=float, default=math.exp(-2.0),
                    help='fit window ends where ISF/ISF(0) first drops below this')
    ap.add_argument('--fit-model', default='single', choices=['single', 'double'])
    ap.add_argument('--fit-skip', type=int, default=0, help='skip first fine-grid points in fits')
    ap.add_argument('--dk-max', type=float, help='largest dK (1/Angstrom)')
    ap.add_argument('--workers', type=int, default=os.cpu_count() or 1)
    ap.add_argument('--backend', default='auto', choices=['auto', 'c', 'numba', 'python'])
    ap.add_argument('--out', default=None, help='output prefix')
    ap.add_argument('--control-file', help='npz of a --control run to overlay on the E_eff plot')
    ap.add_argument('--replot', action='store_true', help='only redo the figures from <out>_results.npz')
    ap.add_argument('--only-temps', type=float, nargs='+',
                    help='run only these temperatures and save raw data to <out>_raw_T<T>.pkl '
                         '(lets long runs be split across jobs); combine with --from-raw')
    ap.add_argument('--from-raw', action='store_true',
                    help='analyse/plot all <out>_raw_T*.pkl files instead of simulating')
    a = ap.parse_args(argv)
    if a.replot:
        prefix = a.out or a.preset
        res = dict(np.load(prefix + '_results.npz'))
        control = dict(np.load(a.control_file)) if a.control_file else None
        print('figures:', make_plots(res, {}, prefix, control))
        return res

    cfg = dict(PRESETS[a.preset])
    for k_cli, k_cfg in [('lattice', 'lattice'), ('coverage', 'coverage'), ('temps', 'temps'),
                         ('Ea', 'Ea'), ('dE', 'dE'), ('lam', 'lam')]:
        if getattr(a, k_cli) is not None:
            cfg[k_cfg] = getattr(a, k_cli)
    if a.L:
        cfg['L'] = (a.L[0], a.L[-1])
    cfg['potential'] = dict(cfg['potential'])
    if a.V0 is not None:
        cfg['potential']['V0'] = a.V0
    if a.potential_json:
        cfg['potential'] = json.loads(a.potential_json)
    if a.control:
        cfg['potential'] = dict(type='none')
    cfg.update(fit_floor=a.fit_floor, fit_model=a.fit_model, fit_skip=a.fit_skip)
    prefix = a.out or (a.preset + ('_control' if a.control else ''))

    lat = make_lattice(cfg['lattice'], a=cfg['a'], b=cfg.get('b'), Ea=cfg['Ea'],
                       Ea2=cfg.get('Ea2'), dE=cfg.get('dE', 0.0))
    pot = build_potential(cfg['potential'], lat)
    L1, L2 = cfg['L']
    tab = build_tables(lat, L1, L2, pot)
    shells, qvecs = build_q_set(lat, L1, L2, a.dk_max)
    # drop dK at which no hop changes rho_q (Bragg-like points: the ISF cannot decay)
    keep = [sh for sh in shells
            if np.max(1 - np.cos(np.einsum('bhx,x->bh', tab['hop_vec'], sh['q']))) > 1e-9]
    newq = []
    for sh in keep:
        ids = list(range(len(newq), len(newq) + len(sh['idx'])))
        newq.extend(qvecs[sh['idx']])
        sh['idx'] = ids
    shells, qvecs = keep, np.array(newq)
    tab['phase'] = np.ascontiguousarray(np.exp(-1j * tab['pos'] @ qvecs.T))
    Np = int(round(cfg['coverage'] * tab['n_cells']))
    be = get_backend(a.backend)
    print(f'lattice={lat.name}  sites={tab["n_sites"]}  particles={Np}  '
          f'theta(per cell)={Np / tab["n_cells"]:.4f}  q-shells={len(shells)}  '
          f'q-vectors={len(qvecs)}  backend={be.name}')
    print(f'potential={cfg["potential"]}  interaction neighbours={tab["n_int"].max()}')

    # time grids from a pilot run at each temperature
    if a.from_raw:
        import glob, pickle
        results, grid = [], {}
        for fn in sorted(glob.glob(prefix + '_raw_T*.pkl')):
            with open(fn, 'rb') as f:
                d = pickle.load(f)
            results += d['results']
            grid.update(d['grid'])
        cfg['grid'] = grid
        print(f'loaded {len(results)} replicas at T = {sorted(grid)}')
        return finish(a, cfg, lat, tab, shells, qvecs, Np, prefix, results, time.time())
    run_temps = a.only_temps if a.only_temps else cfg['temps']
    jobs, grid = [], {}
    min_cos = min(np.mean(1 - np.cos(tab['hop_vec'][:, :tab['n_hop'][0]] @ s['q'])) for s in shells)
    for T in run_temps:
        G = pilot_rate(tab, be, Np, T, cfg['nu0'], cfg['lam'])
        dt = a.dt_factor / G
        a_min = G * min_cos * 0.5                 # conservative slowest decay rate
        nlag_c = int(min(max(64, math.ceil(5.0 / (a_min * dt * a.n_sub))), 8192))
        chunk = max(8192, 1 << int(math.ceil(math.log2(4 * a.nlag)))) - 2 * a.nlag
        grid[T] = dict(dt=dt, n_sub=a.n_sub, Gamma_pilot=G)
        t_eq = max(a.eq_hops / G, 4.0 / a_min)
        n_samples = int(a.prod_hops / (G * dt))
        print(f'  T={T:7.1f} K  Gamma_eff~{G:.4g}/ps  dt={dt:.4g} ps  '
              f'fine window={a.nlag * dt:.4g} ps  coarse window={nlag_c * dt * a.n_sub:.4g} ps  '
              f'samples/replica={n_samples}')
        for s in range(a.n_seeds):
            jobs.append(dict(T=T, seed=int(round(100 * T)) * 1000 + s, Np=Np, nu0=cfg['nu0'], lam=cfg['lam'],
                             dt=dt, n_sub=a.n_sub, nlag_f=a.nlag, nlag_c=nlag_c, chunk=chunk,
                             t_eq=t_eq, n_samples=n_samples))
    cfg['grid'] = grid

    t0 = time.time()
    results = []
    if a.workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(a.workers, initializer=_init_worker,
                                 initargs=(tab, a.backend)) as ex:
            for i, r in enumerate(ex.map(simulate_replica, jobs)):
                results.append(r)
                print(f'  done {i + 1}/{len(jobs)} (T={r["T"]:.0f} K, {r["n_events"]:.3g} events)'
                      f'  {time.time() - t0:.0f} s', flush=True)
    else:
        _init_worker(tab, a.backend)
        for i, j in enumerate(jobs):
            results.append(simulate_replica(j))
            print(f'  done {i + 1}/{len(jobs)} (T={j["T"]:.0f} K, {results[-1]["n_events"]:.3g} events)'
                  f'  {time.time() - t0:.0f} s', flush=True)

    if a.only_temps:
        import pickle
        for T in run_temps:
            with open(f'{prefix}_raw_T{T:g}.pkl', 'wb') as f:
                pickle.dump(dict(results=[r for r in results if r['T'] == T], grid={T: grid[T]}), f)
        print(f'raw data saved for T = {run_temps}  ({time.time() - t0:.0f} s)')
        return None
    return finish(a, cfg, lat, tab, shells, qvecs, Np, prefix, results, t0)


def finish(a, cfg, lat, tab, shells, qvecs, Np, prefix, results, t0):
    res = analyse(results, shells, qvecs, tab, cfg)
    save = {k: v for k, v in res.items() if isinstance(v, np.ndarray)}
    save['az'] = np.array(res['az'])
    save['theta_site'] = np.array(res['theta_site'])
    save['E_Gamma'] = np.array(res['E_Gamma'])
    np.savez(prefix + '_results.npz', **save)
    with open(prefix + '_ISF.npz', 'wb') as f:
        np.savez(f, **{f'{k}_{i}': d[k] for i, d in enumerate(res['ISF']) for k in d})
    summary = dict(preset=a.preset, lattice=lat.name, coverage_per_cell=Np / tab['n_cells'],
                   potential=cfg['potential'], temps=list(map(float, res['temps'])),
                   E_Gamma_meV=float(1e3 * res['E_Gamma']), E_Gamma_err_meV=1e3 * res['E_Gamma_err'],
                   dK=list(map(float, res['dk'])), azimuth=res['az'],
                   E_eff_meV=list(map(float, 1e3 * res['E_alpha'])),
                   E_eff_err_meV=list(map(float, 1e3 * res['E_alpha_err'])),
                   sublattice_fraction=res['sub_frac'].tolist())
    with open(prefix + '_summary.json', 'w') as f:
        json.dump(summary, f, indent=1)
    control = None
    if a.control_file:
        control = dict(np.load(a.control_file))
    figs = make_plots(res, cfg, prefix, control)
    print(f'E_Gamma = {1e3 * res["E_Gamma"]:.2f} +- {1e3 * res["E_Gamma_err"]:.2f} meV')
    for az in dict.fromkeys(res['az']):
        m = np.array([x == az for x in res['az']])
        print(f'{az}: E_eff range {1e3 * np.nanmin(res["E_alpha"][m]):.1f} .. '
              f'{1e3 * np.nanmax(res["E_alpha"][m]):.1f} meV  '
              f'(median error {1e3 * np.nanmedian(res["E_alpha_err"][m]):.2f} meV)')
    print('figures:', ', '.join(figs), f'  total {time.time() - t0:.0f} s')
    return res


if __name__ == '__main__':
    main()
