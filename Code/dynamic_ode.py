r"""
Dynamic Systems and Ordinary Differential Equations (ODE) Solver Module.

Models physical dynamic phenomena associated with packaged meat quality inspection:
    1. Vacuum seal pressure decay:
        \frac{dP}{dt} = -k (P - P_{atm})
    2. Dynamic high-speed sorting reject actuator:
        m \ddot{x} + c \dot{x} + k_s x = F(t)
Implements Explicit Euler, Classical Runge-Kutta 4th Order (RK4), and SciPy RK45,
with temporal convergence and stability checks.
"""

from typing import Tuple, List, Dict
import numpy as np
import scipy.integrate as spi


class DynamicSystemSolver:
    """Solves first- and second-order dynamic engineering models for packaging inspection.

    Attributes:
        k (float): Conductance leak coefficient [s^-1].
        P_vac (float): Initial packaging vacuum pressure [kPa].
        P_atm (float): Ambient atmospheric pressure [kPa].
    """

    def __init__(
        self,
        k_decay: float = 0.8,
        P_vac: float = 20.0,
        P_atm: float = 101.3,
    ) -> None:
        """Initializes the ODE solver with physical parameters.

        Args:
            k_decay: Leak coefficient in s^-1. Defaults to 0.8.
            P_vac: Initial packaged vacuum level in kPa. Defaults to 20.0.
            P_atm: Ambient atmospheric pressure in kPa. Defaults to 101.3.
        """
        self.k = k_decay
        self.P_vac = P_vac
        self.P_atm = P_atm

    def vacuum_rhs(self, t: float, P: float) -> float:
        """Computes the rate of change of vacuum pressure dP/dt.

        Args:
            t: Time [s].
            P: Internal pressure [kPa].

        Returns:
            float: Time derivative dP/dt [kPa/s].
        """
        return -self.k * (P - self.P_atm)

    def analytical_solution(self, t_array: np.ndarray) -> np.ndarray:
        r"""Evaluates the closed-form analytical solution for the vacuum decay IVP.

        P(t) = P_{atm} + (P_{vac} - P_{atm}) \exp(-k t)

        Args:
            t_array: Array of evaluation timestamps [s].

        Returns:
            np.ndarray: True analytical pressure values [kPa].
        """
        return self.P_atm + (self.P_vac - self.P_atm) * np.exp(-self.k * t_array)

    def solve_euler(
        self, t_span: Tuple[float, float] = (0.0, 5.0), dt: float = 0.1
    ) -> Tuple[np.ndarray, np.ndarray]:
        r"""Solves the vacuum decay IVP using Explicit Forward Euler.

        Numerical stability requirement:
            |1 - k \Delta t| < 1  \implies  \Delta t < \frac{2}{k} = 2.5 \text{ s}.

        Args:
            t_span: Tuple of (t0, tf). Defaults to (0.0, 5.0).
            dt: Fixed time step [s]. Defaults to 0.1.

        Returns:
            Tuple of (time_array, pressure_array).
        """
        t0, tf = t_span
        n_steps = int(np.ceil((tf - t0) / dt)) + 1
        t = np.linspace(t0, tf, n_steps)
        P = np.zeros(n_steps, dtype=np.float64)
        P[0] = self.P_vac

        for n in range(n_steps - 1):
            P[n + 1] = P[n] + dt * self.vacuum_rhs(t[n], P[n])

        return t, P

    def solve_rk4(
        self, t_span: Tuple[float, float] = (0.0, 5.0), dt: float = 0.1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Solves the vacuum decay IVP using Classical 4th-Order Runge-Kutta (RK4).

        Args:
            t_span: Tuple of (t0, tf). Defaults to (0.0, 5.0).
            dt: Fixed time step [s]. Defaults to 0.1.

        Returns:
            Tuple of (time_array, pressure_array).
        """
        t0, tf = t_span
        n_steps = int(np.ceil((tf - t0) / dt)) + 1
        t = np.linspace(t0, tf, n_steps)
        P = np.zeros(n_steps, dtype=np.float64)
        P[0] = self.P_vac

        for n in range(n_steps - 1):
            tn = t[n]
            pn = P[n]
            k1 = self.vacuum_rhs(tn, pn)
            k2 = self.vacuum_rhs(tn + 0.5 * dt, pn + 0.5 * dt * k1)
            k3 = self.vacuum_rhs(tn + 0.5 * dt, pn + 0.5 * dt * k2)
            k4 = self.vacuum_rhs(tn + dt, pn + dt * k3)
            P[n + 1] = pn + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        return t, P

    def solve_rk45_adaptive(
        self, t_span: Tuple[float, float] = (0.0, 5.0)
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Solves the IVP using variable-step Runge-Kutta 4(5) (Dormand-Prince).

        Args:
            t_span: Integration range (t0, tf). Defaults to (0.0, 5.0).

        Returns:
            Tuple of (time_array, pressure_array).
        """
        sol = spi.solve_ivp(
            fun=self.vacuum_rhs,
            t_span=t_span,
            y0=[self.P_vac],
            method="RK45",
            rtol=1e-6,
            atol=1e-8,
        )
        return sol.t, sol.y[0]

    def convergence_study(
        self, dt_list: Tuple[float, ...] = (0.5, 0.25, 0.1, 0.05, 0.025, 0.0125)
    ) -> List[Dict[str, float]]:
        """Verifies error scaling orders: O(dt) for Euler, O(dt^4) for RK4.

        Args:
            dt_list: Sequence of temporal step sizes to evaluate.

        Returns:
            List of dictionaries containing dt, Euler RMSE, and RK4 RMSE.
        """
        t_span = (0.0, 5.0)
        records = []
        for dt in dt_list:
            t_eu, p_eu = self.solve_euler(t_span, dt)
            t_rk, p_rk = self.solve_rk4(t_span, dt)

            p_true_eu = self.analytical_solution(t_eu)
            p_true_rk = self.analytical_solution(t_rk)

            rmse_eu = float(np.sqrt(np.mean((p_eu - p_true_eu) ** 2)))
            rmse_rk = float(np.sqrt(np.mean((p_rk - p_true_rk) ** 2)))

            records.append(
                {
                    "dt": dt,
                    "rmse_euler": rmse_eu,
                    "rmse_rk4": rmse_rk,
                }
            )
        return records

    def reject_actuator_system(
        self,
        m: float = 1.2,
        c: float = 15.0,
        k_s: float = 250.0,
        f0: float = 50.0,
        pulse_width: float = 0.08,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Solves 2nd-order dynamic sorting reject arm:
        State vector: [x, v], where x = displacement (m), v = velocity (m/s).
        m * dv/dt = F(t) - c*v - k_s*x

        Args:
            m: Actuator mass [kg]. Defaults to 1.2.
            c: Viscous damping coefficient [N s / m]. Defaults to 15.0.
            k_s: Return spring stiffness [N / m]. Defaults to 250.0.
            f0: Solenoid peak force [N]. Defaults to 50.0.
            pulse_width: Actuation pulse duration [s]. Defaults to 0.08.

        Returns:
            Tuple of (time_array, displacement_m, velocity_m_s).
        """
        def rhs(t: float, state: List[float]) -> List[float]:
            x, v = state
            f_applied = f0 if t <= pulse_width else 0.0
            dxdt = v
            dvdt = (f_applied - c * v - k_s * x) / m
            return [dxdt, dvdt]

        t_span = (0.0, 0.5)
        sol = spi.solve_ivp(rhs, t_span, [0.0, 0.0], method="RK45", rtol=1e-6, atol=1e-8)
        return sol.t, sol.y[0], sol.y[1]
