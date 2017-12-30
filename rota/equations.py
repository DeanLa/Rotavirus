from collections import namedtuple

import numpy as np
# from .model import Rota, RotaData
from rota import *

COMP = namedtuple("Compartments",
                  '''M S1 Ia1 Im1 Is1 R1 S2 Ia2 Im2 Is2 R2 S3 Ia3 Im3 Is3 R3 
                  V1 V2 V3 Iav Imv Isv''')


def collect_state_0(fixed: RotaData):
    dist = fixed.age_dist
    M = 0.01
    R = 0.75
    S = 0.2
    I = 1 - M - S - R
    c = COMP._make(dist * 0 for _ in COMP._fields)
    assert type(c) == COMP
    c.M[:] += dist * M
    c.R1[:] = dist * R * 3 / 45
    c.R2[:] = dist * R * 18 / 45
    c.R3[:] = dist * R * 21 / 45
    c.S1[:] = dist * S / 3
    c.S2[:] = dist * S / 3
    c.S3[:] = dist * S / 3
    c.Ia1[:] = dist * I / 9
    c.Im1[:] = dist * I / 9
    c.Is1[:] = dist * I / 9
    c.Ia2[:] = dist * I / 9
    c.Im2[:] = dist * I / 9
    c.Is2[:] = dist * I / 9
    c.Ia3[:] = dist * I / 9
    c.Im3[:] = dist * I / 9
    c.Is3[:] = dist * I / 9
    # ret = COMP._make([dist * v for v in c])
    return c


def rota_eq(mcmc, fixed_data: RotaData, start=None, end=None, state_0=None):
    m = mcmc
    if start is None: start = m.start
    if end is None: end = m.end
    if state_0 is None: state_0 = m.state_0
    d = RotaData()
    no_likelihood = ''
    timeline = np.arange(start, end, d.N)
    num_steps = len(timeline)
    c: COMP = COMP._make([np.zeros((d.J, num_steps)) for _ in COMP._fields])
    iter_comps = range(len(c))
    for i in iter_comps:
        c[i][:, 0] = state_0[i]
    for t, T in enumerate(timeline[1:], start=1):
        n: COMP = COMP._make([np.maximum(0, comp[:, t - 1]) for comp in c])
        for i in iter_comps:
            c[i][:, t] = c[i][:, t - 1]
        nI = d.psia1 * n.Ia1 + d.psim1 * n.Im2 + d.psis1 + n.Is1 + \
             d.psia2 * n.Ia2 + d.psim2 * n.Im2 + d.psis2 + n.Is2 + \
             d.psia2 * n.Ia3 + d.psim2 * n.Im3 + d.psis2 + n.Is3
        IC = nI.dot(d.C)
        lamda = 0
        # Start Equations
        # Maternal Immunity
        c.M[0, t] += d.delta
        c.M[:, t] -= d.omega0 * c.M[:, t - 1]  # OUT: Waning to S1

        # Susceptible
        c.S1[:, t] += d.omega0 * c.M[:, t - 1]  # IN: Waning from M
        c.S2[:, t] += d.omega1 * c.R1[:, t - 1] # IN: Waning from R1
        c.S3[:, t] += d.omega2 * c.R2[:, t - 1] # IN: Waning from R2
        c.S3[:, t] += d.omega3 * c.R3[:, t - 1] # IN: Waning from R3


        # Aging
        # c.M[:-1, t] -= d.d[:-1] * c.M[:-1, t]  # OUT
        # c.M[1:, t] += d.d[:-1] * c.M[:-1, t]  # IN

        # Death and Aging
        for i in iter_comps:
            c[i][:-1, t] -= d.d[:-1] * c[i][:-1, t]  # OUT
            c[i][1:, t] += d.d[:-1] * c[i][:-1, t]  # IN
            c[i][:, t] -= d.mu * c[i][:, t - 1] # Death

    return c