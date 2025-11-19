import numpy as np
from typing import Tuple

"""
Core module for Inertial Reference Systems (IRS). 
For simplicity, all velocities are in natural units (c=1).
Provides basic relativity functions and IRS Class.

These functions are intended for implemententation in the main code as the mathematical 
and physical foundation for plotting Minkowski 1+1D diagrams.
"""

def gamma(v: float):
        """Lorentz factor for velocity v (|v| < 1)."""
        if abs(v) >= 1:
                raise ValueError
        return 1 / np.sqrt(1 - v**2)

def lorentz_transform(t_rest: float, x_rest: float, v: float) -> Tuple[float, float]:
        """Transform coordinates from the rest frame to a moving frame."""
        t_prime = gamma(v) * (t_rest - v*x_rest)
        x_prime = gamma(v) * (x_rest - v*t_rest)
        return t_prime, x_prime

def inverse_lorentz_transform(t_prime: float, x_prime: float, v: float) -> Tuple[float, float]:
        """Transform coordinates from a moving frame back to the rest frame."""
        t_rest = gamma(v) * (t_prime + v*x_prime)
        x_rest = gamma(v) * (x_prime + v*t_prime)
        return t_rest, x_rest