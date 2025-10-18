import numpy as np
import scipy




class Solver:

    def __init__(self, system, initial_conditions,T=15, fps=60 ,t_span=None, t_eval=None):
        self.system = system
        self.initial_cond = initial_conditions

        self.T = T
        self.fps = fps

        self.t_span = t_span or [0, T]
        self.t_eval = t_eval or np.linspace(0, T, int(T*fps)+1)

    def run(self): # RETURNS STATE VARIABLE
        try:
            sol = scipy.integrate.solve_ivp(self.system.dynamics, t_span=self.t_span,t_eval=self.t_eval, y0= self.initial_cond, method="DOP853", rtol=1e-6, atol=1e-8)

        except Exception as e:
            print("[Solver Error] Integration failed:", e)
            raise

        if not sol.success:
            print(f"[Solver Warning] Integration unsuccessful: {sol.message}")

        if np.any(np.isnan(sol.y)) or np.any(np.isinf(sol.y)):
            raise FloatingPointError("[Solver Error] NaN or Inf detected in simulation result.")

        return sol

