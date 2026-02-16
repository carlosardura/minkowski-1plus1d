import numpy as np
from typing import Tuple

class MinkowskiEngine:
    def __init__(self):
        self.rest = np.empty((2,0))
        self.names = []
        self.metric = np.diag([1, -1])   # Minkowski metric tensor

    def add_event(self, t: float, x: float, name: str):
        new_event = np.array([[t], [x]])
        self.rest = np.hstack([self.rest, new_event])
        self.names.append(name)

    @staticmethod
    def lorentz_matrix(v: float) -> np.ndarray:
        if abs(v) >= 1: raise ValueError
        gamma = 1 / np.sqrt(1 - v**2)   # Lorentz factor
        return np.array([[gamma, -v*gamma], [-v*gamma, gamma]])


class Event:
    def __init__(self, engine: MinkowskiEngine, index: int):
        self.engine = engine
        self.index = index

    @property
    def name_base(self) -> str:
        return self.engine.names[self.index]

    def coordinates(self, v = 0.0) -> np.ndarray:
        coords_rest = self.engine.rest[:, [self.index]]
        if v == 0: return coords_rest
        return self.engine.lorentz_matrix(v) @ coords_rest

    def causal_structure(self, epsilon = 1e-9) -> Tuple[float, str]:
        x = self.engine.rest[:, [self.index]]
        s2 = (x.T @ self.engine.metric @ x).item()
        if s2 > epsilon: causality = "timelike"
        elif s2 < -epsilon: causality = "spacelike"
        else: causality = "lightlike"
        return (s2, causality)