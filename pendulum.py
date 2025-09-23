import numpy as np



class Pendulum:
    def __init__(self, length, mass, mode, damping=0.0, gravity=9.81, mass_model="point", I=None, lc=None):
        self.l = length
        self.m = mass
        self.b = damping
        self.g = gravity
        self.mass_model = mass_model


        #Sprawdzanie jaki moment bezwładności wybrać
        if mass_model == "point":
            moment_of_inertia = self.m * self.l**2
            mass_center_distance = self.l
        elif mass_model == "uniform":
            moment_of_inertia = (1.0/3.0) * (self.m *self.l**2)
            mass_center_distance = 0.5 * self.l
        else:
            raise ValueError(f"Unknown mass_model: {mass_model}")

        self.I = moment_of_inertia if I is None else I   # I to bezwładność (moment of inertia)
        self.lc = mass_center_distance if lc is None else lc  # lc to odległość do środka masy (length to center)



        self.mode = mode
        self.mode_map = {
            "damped": self.dynamics_damped,
            "ideal": self.dynamics_ideal
        }
        if self.mode not in self.mode_map:
            raise ValueError(f"Unknown mode {self.mode}")


    #Funkcja która za pomocą self.mode uruchamia odpowiednią funkcję pasującą do tego trybu
    def dynamics(self, t, state):
        return self.mode_map[self.mode](t, state)


    def dynamics_ideal(self, t, state):

        theta, theta_dot = state  # theta = angle, theta_dot = angular velocity
        theta_dt = theta_dot
        dtheta_dt = -(self.m * self.g * self.lc)/self.I * np.sin(theta) # po skróceniu wychodzi wzór -g/l *sin(theta), I=ml^2

        return np.array([theta_dt, dtheta_dt])

    def dynamics_damped(self, t, state):

        theta, theta_dot = state  # theta = angle, theta_dot = angular velocity
        theta_dt = theta_dot

        dtheta_dt = (-self.b * theta_dot - self.m *self.g * self.lc * np.sin(theta))/self.I

        return np.array([theta_dt, dtheta_dt])


    def positions(self, state):
        theta, theta_dot = state
        x_tip = self.l * np.sin(theta)
        y_tip = - self.l * np.cos(theta)
        x_center = self.lc * np.sin(theta)
        y_center = - self.lc * np.cos(theta)

        """ 
        Possible change that to include more options for visualizer to take like type of blob to draw on which point
        """
        return [(0, 0),(x_center, y_center), (x_tip, y_tip)] # Wymaga poprawy w visualizerze


    #Funkcja sprawdza energie kinetyczną i potencjalną
    def energy_check(self, state):
        theta, theta_dot = state

        kinetic_energy = 0.5 * self.I * theta_dot**2
        potential_energy = self.m * self.g * self.lc * (1 - np.cos(theta))


        # OLD CODE bez uwzględnienia momentu bezwładności
        """x_dot = self.l * theta_dot * np.cos(theta)
        y_dot = self.l * theta_dot * np.sin(theta)

        # Total kinetic energy
        kinetic_energy = self.m * (x_dot**2 + y_dot**2)/2

        # Total potential energy
        h = self.l * (1 - np.cos(theta))
        potential_energy = self.m * self.g * h
        """

        total_energy = kinetic_energy + potential_energy

        return np.array([kinetic_energy, potential_energy, total_energy])