import numpy as np


class InvertedPendulum:
    def __init__(self, mode, l=0.3, m=0.2, cart_mass=0.5, g=9.81, mass_model="point", I=None):
        self.l = l
        self.m = m
        self.g = g
        self.M = cart_mass

        self.mass_model = mass_model

        if self.mass_model == "point":
            moment_of_inertia = self.m * self.l**2
            mass_center_distance = self.l
        elif self.mass_model == "unifrom":
            moment_of_inertia = (1.0/3.0) * (self.m * self.l**2)
            mass_center_distance = 0.5 * self.l

        self.I = moment_of_inertia
        self.l_mc = mass_center_distance

        self.mode = mode
        self.mode_map = {
            "ideal" : self.dynamics_ideal
        }
        if self.mode not in self.mode_map:
            raise ValueError(f"Unknown mode {self.mode}")


    def dynamics_ideal(self, t, u, state): # u is the cart force (F)
        x, x_dot, theta, theta_dot = state  # x is position of the cart, x_dot is velocity of the cart, # theta_dot is angular velocity
        m, M, g = self.m, self.M, self.g
        l_mc, I = self.l_mc, self.I

        divisor = (M + m) * (I + m*l_mc**2) - (m * l_mc * np.cos(theta))**2

        x_double_dot = (( (I + m*l_mc**2) * (u + m*l_mc*np.sin(theta)*theta_dot**2) - m**2 *l_mc**2*g*np.sin(theta)*np.cos(theta))
                        / divisor)

        theta_double_dot = ( -(m*l_mc*np.cos(theta)) * (u+m*l_mc*np.sin(theta)*theta_dot**2) + ((M+m)*m*g*l_mc*np.sin(theta)) )/divisor



        return np.array([x_dot, x_double_dot, theta_dot, theta_double_dot], dtype=float)

    def dynamics_damped(self, t, u, state,
                        b_cart=0.0, b_pend=0.0, Fc_cart=0.0, Tc_pivot=0.0):

        x, x_dot, theta, theta_dot = state
        m, M, g = self.m, self.M, self.g
        l_mc, I = self.l_mc, self.I

        """
        Coulomb is 0 at zero velocity for stability???? 
        """
        sign = lambda v: np.tanh(1e3 * v) # idk why to use lambda or tanh
        u_eff = u - b_cart * x_dot - Fc_cart * sign(x_dot)

        divisor = (M + m) * (I + m * l_mc) - (m * l_mc * np.cos(theta)) ** 2

        rhs1 = u_eff + m * l_mc * np.sin(theta) * theta_dot ** 2
        rhs2 = m * g * l_mc * np.sin(theta) - b_pend * theta_dot - Tc_pivot * sign(theta_dot)

        x_double_dot = ((I + m * l_mc ** 2) * rhs1 - (m * l_mc * np.cos(theta)) * rhs2) / divisor
        theta_double_dot = (-(m * l_mc * np.cos(theta)) * rhs1 + (M + m) * rhs2) / divisor


        return np.array([x_dot, x_double_dot, theta_dot, theta_double_dot], dtype=float)


    def positions(self, state):
        x, xdot, theta, thetadot = state
        pivot = np.array([x, 0.0])  # cart’s pivot on the track
        tip = pivot + np.array([self.l * np.sin(theta), self.l * np.cos(theta)])
        return np.vstack([pivot, tip], dtype=float)  # shape (2, 2): [ (x0,y0), (x1,y1) ]

    def energy_check(self, state):
        x, x_dot, theta, theta_dot = state
        m, M, g, l = self.m, self.M, self.g, self.l

        # kinetic
        cart_kinetic = 0.5 * M * x_dot ** 2
        pend_kinetic = 0.5 * m * (
                (x_dot + l * theta_dot * np.cos(theta)) ** 2 +
                (l * theta_dot * np.sin(theta)) ** 2
        )
        kinetic_energy = pend_kinetic

        # potential (zero at pendulum state=[0, 0, pi, 0])
        y_pend = l * np.cos(theta) + l
        potential_energy = m * g * y_pend

        total_energy = kinetic_energy + potential_energy

        return np.array([kinetic_energy, potential_energy, total_energy])

