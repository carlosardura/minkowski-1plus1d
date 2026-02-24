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
    
    def boost(self, x: np.ndarray, v: float) -> np.ndarray:
        if v == 0: return x
        return self.lorentz_matrix(v) @ x
    
    def causal_structure(self, x: np.ndarray, epsilon = 1e-9) -> Tuple[float, str]:
        s2 = (x.T @ self.metric @ x).item()
        if s2 > epsilon: causality = "timelike"
        elif s2 < -epsilon: causality = "spacelike"
        else: causality = "lightlike"
        return (s2, causality)


class Event:
    def __init__(self, engine: MinkowskiEngine, index: int):
        self.engine = engine
        self.index = index

    @property
    def name_base(self) -> str:
        return self.engine.names[self.index]

    def coordinates(self, v: float = 0.0) -> np.ndarray:
        x_rest = self.engine.rest[:, [self.index]]
        return self.engine.boost(x_rest, v)

    def causality(self) -> Tuple[float, str]:
        x = self.engine.rest[:, [self.index]]
        return self.engine.causal_structure(x)


class Segment:
    def __init__(self, engine: MinkowskiEngine, index1: int, index2: int):
        self.engine = engine
        self.index1 = index1
        self.index2 = index2

    def coordinates(self, v: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        x1_rest = self.engine.rest[:, [self.index1]]
        x2_rest = self.engine.rest[:, [self.index2]]
        return (self.engine.boost(x1_rest, v), self.engine.boost(x2_rest, v))

    def causality(self) -> Tuple[float, str]:
        dx = self.engine.rest[:, [self.index2]] - self.engine.rest[:, [self.index1]]
        return self.engine.causal_structure(dx)