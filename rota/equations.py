# from rota import *
from collections import namedtuple
import numpy as np

# from rota import RotaData
from rota import COMP, logger, RotaData, concat_nums


def collect_state_0(fixed):
    dist = fixed().age_dist
    M = 0.0025
    R = 0.8
    S = 0.1
    I = 1 - M - S - R
    c = COMP._make(dist * 0 for _ in COMP._fields)
    assert type(c) == COMP
    c.M[:] = dist * M
    c.R1[:] = dist * R * 0.025
    c.R2[:] = dist * R * 0.025
    c.R3[:] = dist * R * 0.95
    c.S1[:] = dist * S * 0.01
    c.S2[:] = dist * S * 0.01
    c.S3[:] = dist * S * 0.99
    c.Ia1[:] = dist * I / 9
    c.Im1[:] = dist * I / 9
    c.Is1[:] = dist * I / 9
    c.Ia2[:] = dist * I / 9
    c.Im2[:] = dist * I / 9
    c.Is2[:] = dist * I / 9
    c.Ia3[:] = dist * I / 9
    c.Im3[:] = dist * I / 9
    c.Is3[:] = dist * I / 9
    # print(sum(c))
    # print(sum(c).sum())
    return c


def state_z_to_state_0(c: COMP):
    return COMP._make([comp[:, -1] for comp in c])


def seasonal(t, A, offset=0):
    return 1 + A * np.cos(2 * np.pi * (t - offset))


def rota_eq(mcmc, steps=None, start=None, end=None, state_0=None):
    logger.info("Running Model")
    low_force = 10e-8
    m = mcmc
    d = RotaData(steps)
    if start is None:
        start = m.start
    if end is None:
        end = m.end
    if state_0 is None:
        state_0 = m.state_0
    timeline = np.arange(start, end, d.N)
    num_steps = len(timeline)
    c: COMP = COMP._make([np.zeros((d.J, num_steps)) for _ in COMP._fields])
    iter_comps = range(len(c))
    for i in iter_comps:
        c[i][:, 0] = state_0[i]
    sa = d.age_union
    b = concat_nums(m.b1, sa[0], m.b2, sa[1], m.b3, sa[2], m.b4, sa[3], m.b5, sa[4]) * 1e-7
    assert len(b) == d.J
    # print (b)
    for t, T in enumerate(timeline[1:], start=1):
        # print(t, '--------')
        n: COMP = COMP._make([np.maximum(0, comp[:, t - 1]) for comp in c])
        # print("n", sum(n).sum())
        for i in iter_comps:
            c[i][:, t] = c[i][:, t - 1]
        nI = d.psia1 * n.Ia1 + d.psim1 * n.Im2 + d.psis1 + n.Is1 + \
             d.psia2 * n.Ia2 + d.psim2 * n.Im2 + d.psis2 + n.Is2 + \
             d.psia3 * n.Ia3 + d.psim3 * n.Im3 + d.psis3 + n.Is3
        IC = nI.dot(d.C)
        lamda = b * IC * seasonal(T, m.A / 10, m.offset / 10) + low_force
        # Start Equations
        # Maternal Immunity
        c.M[0, t] += d.delta
        c.M[:, t] -= d.omega0 * c.M[:, t - 1]  # OUT: Waning to S1

        # Susceptible
        c.S1[:, t] += d.omega0 * c.M[:, t - 1]  # IN: Waning from M
        c.S2[:, t] += d.omega1 * c.R1[:, t - 1]  # IN: Waning from R1
        c.S3[:, t] += d.omega2 * c.R2[:, t - 1]  # IN: Waning from R2
        c.S3[:, t] += d.omega3 * c.R3[:, t - 1]  # IN: Waning from R3

        c.S1[:, t] -= d.phi1 * lamda * c.S1[:, t - 1]  # OUT: Getting Sick
        c.S2[:, t] -= d.phi2 * lamda * c.S2[:, t - 1]  # OUT: Getting Sick
        c.S3[:, t] -= d.phi3 * lamda * c.S3[:, t - 1]  # OUT: Getting Sick

        # Infected
        I1 = d.phi1 * lamda * c.S1[:, t - 1]  # Helper I
        I2 = d.phi2 * lamda * c.S2[:, t - 1]  # Helper I
        I3 = d.phi3 * lamda * c.S3[:, t - 1]  # Helper I
        c.Ia1[:, t] += d.rhoa1 * I1  # IN: Infected from S1
        c.Im1[:, t] += d.rhom1 * I1  # IN: Infected from S1
        c.Is1[:, t] += d.rhos1 * I1  # IN: Infected from S1
        c.Ia2[:, t] += d.rhoa2 * I2  # IN: Infected from S2
        c.Im2[:, t] += d.rhom2 * I2  # IN: Infected from S2
        c.Is2[:, t] += d.rhos2 * I2  # IN: Infected from S2
        c.Ia3[:, t] += d.rhoa3 * I3  # IN: Infected from S3
        c.Im3[:, t] += d.rhom3 * I3  # IN: Infected from S3
        c.Is3[:, t] += d.rhos3 * I3  # IN: Infected from S3

        c.Ia1[:, t] -= d.gammaa1 * c.Ia1[:, t - 1]  # Recovered to1
        c.Im1[:, t] -= d.gammam1 * c.Im1[:, t - 1]  # Recovered to1
        c.Is1[:, t] -= d.gammas1 * c.Is1[:, t - 1]  # Recovered to1
        c.Ia2[:, t] -= d.gammaa2 * c.Ia2[:, t - 1]  # Recovered to2
        c.Im2[:, t] -= d.gammam2 * c.Im2[:, t - 1]  # Recovered to2
        c.Is2[:, t] -= d.gammas2 * c.Is2[:, t - 1]  # Recovered to2
        c.Ia3[:, t] -= d.gammaa3 * c.Ia3[:, t - 1]  # Recovered to3
        c.Im3[:, t] -= d.gammam3 * c.Im3[:, t - 1]  # Recovered to3
        c.Is3[:, t] -= d.gammas3 * c.Is3[:, t - 1]  # Recovered to3

        # Vaccinated
        c.V1[:, t] -= lamda * c.V1[:, t - 1] * d.reduct1  # OUT: Getting Sick
        c.V2[:, t] -= lamda * c.V2[:, t - 1] * d.reduct2  # OUT: Getting Sick
        c.V3[:, t] -= lamda * c.V3[:, t - 1] * d.reduct3  # OUT: Getting Sick

        # Infected from V
        Iv1 = lamda * c.V1[:, t - 1] * d.reduct1  # Helper I
        Iv2 = lamda * c.V2[:, t - 1] * d.reduct2  # Helper I
        Iv3 = lamda * c.V3[:, t - 1] * d.reduct3  # Helper I
        c.Iav[:, t] += d.rhoav1 * Iv1  # IN: Infected from S1
        c.Imv[:, t] += d.rhomv1 * Iv1  # IN: Infected from S1
        c.Isv[:, t] += d.rhosv1 * Iv1  # IN: Infected from S1
        c.Iav[:, t] += d.rhoav2 * Iv2  # IN: Infected from S2
        c.Imv[:, t] += d.rhomv2 * Iv2  # IN: Infected from S2
        c.Isv[:, t] += d.rhosv2 * Iv2  # IN: Infected from S2
        c.Iav[:, t] += d.rhoav3 * Iv3  # IN: Infected from S3
        c.Imv[:, t] += d.rhomv3 * Iv3  # IN: Infected from S3
        c.Isv[:, t] += d.rhosv3 * Iv3  # IN: Infected from S3

        c.Iav[:, t] -= d.gammaav * c.Iav[:, t - 1]  # Recovered to R1
        c.Imv[:, t] -= d.gammamv * c.Imv[:, t - 1]  # Recovered to R1
        c.Isv[:, t] -= d.gammasv * c.Isv[:, t - 1]  # Recovered to R1

        # Recovered
        c.R1[:, t] -= d.omega1 * c.R1[:, t - 1]  # OUT: Waning to S2
        c.R2[:, t] -= d.omega2 * c.R2[:, t - 1]  # OUT: Waning to S3
        c.R3[:, t] -= d.omega3 * c.R3[:, t - 1]  # OUT: Waning to S3

        c.R1[:, t] += d.gammaa1 * c.Ia1[:, t - 1]  # Recovered from Ia1
        c.R1[:, t] += d.gammam1 * c.Im1[:, t - 1]  # Recovered from Im1
        c.R1[:, t] += d.gammas1 * c.Is1[:, t - 1]  # Recovered from Is1
        c.R2[:, t] += d.gammaa2 * c.Ia2[:, t - 1]  # Recovered from Ia2
        c.R2[:, t] += d.gammam2 * c.Im2[:, t - 1]  # Recovered from Im2
        c.R2[:, t] += d.gammas2 * c.Is2[:, t - 1]  # Recovered from Is2
        c.R3[:, t] += d.gammaa3 * c.Ia3[:, t - 1]  # Recovered from Ia3
        c.R3[:, t] += d.gammam3 * c.Im3[:, t - 1]  # Recovered from Im3
        c.R3[:, t] += d.gammas3 * c.Is3[:, t - 1]  # Recovered from Is3
        # From Iv
        c.R1[:, t] += d.gammaav * c.Iav[:, t - 1]  # Recovered from Iav
        c.R1[:, t] += d.gammamv * c.Imv[:, t - 1]  # Recovered from Imv
        c.R1[:, t] += d.gammasv * c.Isv[:, t - 1]  # Recovered from Isv

        # Death and Aging
        for i in iter_comps:
            c[i][:-1, t] -= d.d[:-1] * c[i][:-1, t - 1]  # OUT
            c[i][1:, t] += d.d[:-1] * c[i][:-1, t - 1]  # IN
            c[i][:, t] -= d.mu * c[i][:, t - 1]  # Death

        if T >= 9:
            # Pass coverage rate to Vaccine Groupd
            c.M[1, t] -= d.d[0] * c.M[0, t - 1] * d.cover1
            c.S1[1, t] -= d.d[0] * c.S1[0, t - 1] * d.cover1
            c.V1[1, t] += d.d[0] * c.M[0, t - 1] * d.cover1
            c.V1[1, t] += d.d[0] * c.S1[0, t - 1] * d.cover1

            c.V1[2, t] -= d.d[1] * c.V1[1, t - 1] * d.cover2
            c.V2[2, t] += d.d[1] * c.V2[1, t - 1] * d.cover2
            c.V2[3, t] -= d.d[2] * c.V2[2, t - 1] * d.cover3
            c.V3[3, t] += d.d[2] * c.V3[2, t - 1] * d.cover3

        pop = sum([comp[:, t] for comp in c]).sum()
        if (pop < -0.5):
            logger.error('explode equations {}'.format(pop))
            raise ArithmeticError
        # A = [comp[:, t] for comp in c]
        # print (sum(A))
        # print("A", sum(A).sum())
        # if sum(A).sum() is np.nan:
        #     print("OH NO")
        #     raise AssertionError
        # print()
    return c
