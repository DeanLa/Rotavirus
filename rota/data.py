import numpy as np


class ClinicalData(object):
    pass


class RotaData(ClinicalData):
    JAPAN_POPULATION = 127_000_000
    long_infection_duration = 7  # days
    short_infection_duration = 3.5  # days
    age_union = np.array([4, 1, 3, 1, 5])
    ages = np.array((0, 2 / 12, 4 / 12, 6 / 12, 1, 2, 3, 4, 5, 15, 20, 30, 50, 65, 100))

    def __init__(self, steps_in_year=None):
        self.steps = steps_in_year if steps_in_year else 52
        self.N = 1 / self.steps
        # Age
        self.a_u = self.ages[1:]
        self.a_l = self.ages[:-1]
        self.a = (self.a_u - self.a_l)
        # self.age_dist = self.a / self.a.sum()
        self.J = len(self.a)  # num age groups


        # Death Rate
        self.mu = np.array((3, 3, 3, 3,
                            0.5, 0.25, 0.251, 0.251,
                            0.1, 0.1, 0.381835, 0.8125,
                            3.558, 22.2)) / 1000
        # Age transition
        self.d = self.mu / (np.e ** (self.mu * self.a) - 1)
        self.d[-1] = 0
        mu_d = self.mu + self.d
        tmp = self.d[:-1] / mu_d[1:]
        self.delta = (mu_d[0]) / (1 + tmp.cumprod().sum())

        n0 = self.delta / mu_d[0]
        age_dist = np.append(n0, np.zeros((self.J - 1)))
        for i in range(1, self.J):
            age_dist[i] = self.d[i - 1] * age_dist[i - 1] / mu_d[i]
        self.age_dist = age_dist

        # Set to relevant rate
        self.mu *= self.N
        self.d *= self.N
        self.delta *= self.N

        # Contact Mixing
        self.C = 365 * np.genfromtxt('./data/contact-mixing.csv', delimiter='\t', dtype=np.float32) * self.N

        # Probability of severity given infection
        self.rhoa1, self.rhom1, self.rhos1 = 0.53, 0.34, 0.13
        self.rhoa2, self.rhom2, self.rhos2 = 0.75, 0.22, 0.03
        self.rhoa3, self.rhom3, self.rhos3 = 0.8, 0.2, 0.0

        # Duration of infection (days)
        high = self.N * (365 / self.long_infection_duration)
        low = self.N * (365 / self.short_infection_duration)
        self.gammaa1 = low
        self.gammam1 = high
        self.gammas1 = high
        self.gammaa2 = low
        self.gammam2 = low
        self.gammas2 = low
        self.gammaa3 = low
        self.gammam3 = low
        self.gammas3 = low

        # Relative infectiousness
        self.psia1, self.psim1, self.psis1 = 0.1, 1.0, 1.0
        self.psia2, self.psim2, self.psis2 = 0.1, 0.5, 0.8
        self.psia3, self.psim3, self.psis3 = self.psia2, self.psim2, self.psis2

        # Relative susceptability following:
        self.phi1 = 1  # Primarty infection
        self.phi2 = 0.62  # Second and subsequent infection
        self.phi3 = 0.37

        # Duration of immunity (Months)
        self.omega0 = self.N * (12 / 3)
        self.omega1 = self.N * (12 / 18)
        self.omega2 = self.N * (12 / 24)
        self.omega3 = self.omega2

        # State_0

        # self.age_dist = self.a / self.a.sum()
        #

        # helper = 0
        # for i in range(1, self.J):
        #     h = 1
        #     for j in range(1, i + 1):
        #         h *= (self.d[j - 1] / mu_d[j])
        #     helper += h
        # mu_d[0] / (1 + helper)


if __name__ == '__main__':
    d = RotaData()
    print(d.C)
