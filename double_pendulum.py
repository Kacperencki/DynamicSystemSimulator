import numpy as np


class DoublePendulum:

    def __init__(self, mass1, length1, mass2, length2, initial_conditons, gravity=9.81):
        self.m1 = mass1
        self.m2 = mass2
        self.l1 = length1
        self.l2 = length2
        self.initial_cond = initial_conditons # [theta1, dtheta1, theta2, dtheta2]
        self.g = gravity

    def dynamics(self, y):
        theta1, dtheta1, theta2, dtheta2 = y # Muszę zastąpić y czymś bardziej przydatnym

        delta = theta1 - theta2

        denom = 2*self.m1 + self.m2 - self.m2 * np.cos(2 * delta)

        theta1_dt = dtheta1
        theta2_dt = dtheta2

        dtheta1_dt = (
            -self.g * (2 * self.m1 + self.m2) * np.sin(theta1)
            - self.m2 * self.g * np.sin(theta1-2*theta2)
            - 2 * np.sin(delta) * self.m2 * (dtheta2**2 * self.l2 + dtheta1**2 * self.l1 * np.cos(delta))
        ) / (self.l1 * denom)

        dtheta2_dt = (
            2 * np.sin(delta) * (
                dtheta1**2 * self.l1 * (self.m1 + self.m2)
                + self.g * (self.m1 + self.m2) * np.cos(theta1)
                + dtheta2**2 * self.l2 * self.m2 * np.cos(delta)
            )
        ) / (self.l2 * denom)

        return [theta1_dt, dtheta1_dt, theta2_dt, dtheta2_dt]


