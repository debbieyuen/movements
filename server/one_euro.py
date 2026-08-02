"""One Euro filter (Casiez et al. 2012): jitter-free at rest, responsive in
motion. Applied per channel to landmark coordinates before IK.

min_cutoff: baseline smoothing (lower = smoother at rest)
beta:       speed coefficient (higher = snappier during fast motion)
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np


class OneEuro:
    def __init__(self, min_cutoff: float = 1.5, beta: float = 0.3,
                 d_cutoff: float = 1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self._x_prev: Optional[np.ndarray] = None
        self._dx_prev: Optional[np.ndarray] = None
        self._t_prev: Optional[float] = None

    @staticmethod
    def _alpha(cutoff: float | np.ndarray, dt: float) -> float | np.ndarray:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def reset(self) -> None:
        self._x_prev = self._dx_prev = self._t_prev = None

    def filter(self, x: np.ndarray, t: float) -> np.ndarray:
        """Filter a sample `x` (any shape) taken at time `t` seconds."""
        x = np.asarray(x, dtype=np.float64)
        if self._x_prev is None or self._t_prev is None or t <= self._t_prev:
            self._x_prev = x.copy()
            self._dx_prev = np.zeros_like(x)
            self._t_prev = t
            return x.copy()

        dt = t - self._t_prev
        dx = (x - self._x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx_prev

        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1.0 - a) * self._x_prev

        self._x_prev, self._dx_prev, self._t_prev = x_hat, dx_hat, t
        return x_hat.copy()
