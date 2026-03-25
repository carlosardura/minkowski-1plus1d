import numpy as np
from typing import Tuple, List

class MinkowskiEngine:
    def __init__(self):
        self.rest = np.empty((2,0))
        self.names = []
        self.metric = np.diag([1, -1])   # Minkowski metric tensor
        self.e0 = np.array([[1],[0]])   # unit time vector at rest
        self.e1 = np.array([[0],[1]])   # unit space vector at rest

    @property
    def light_vectors(self) -> np.ndarray:
        return np.array([[1.0, 1.0], [1.0, -1.0]])

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

    def coordinate_deltas(self, v: float = 0.0) -> Tuple[float, float]:
        p1, p2 = self.coordinates(v)
        dt_prime = p2[0, 0] - p1[0, 0]
        dx_prime = p2[1, 0] - p1[1, 0]
        return dt_prime, dx_prime

    def causality(self) -> Tuple[float, str]:
        dx = self.engine.rest[:, [self.index2]] - self.engine.rest[:, [self.index1]]
        return self.engine.causal_structure(dx)


class ReferenceFrame:
    def __init__(self, engine: MinkowskiEngine, v: float, index: int, colors: Tuple[str, str]):
        self.engine = engine
        self.v = v
        self.index = index
        self.colors = colors
        self.is_active = False

    @property
    def color(self) -> str:
        return self.colors[0] if self.is_active else self.colors[1]

    @property
    def label(self) -> str:
        return "S" if self.index == 0 else f"S{self.index}"
    
    def axes(self) -> Tuple[np.ndarray, np.ndarray]:
        t_axis = self.engine.boost(self.engine.e0, -self.v)
        x_axis = self.engine.boost(self.engine.e1, -self.v)
        return t_axis, x_axis


class FrameManager:
    tab20 = [
        ("#1f77b4", "#aec7e8"), ("#ff7f0e", "#ffbb78"),
        ("#2ca02c", "#98df8a"), ("#d62728", "#ff9896"),
        ("#9467bd", "#c5b0d5"), ("#8c564b", "#c49c94"),
        ("#e377c2", "#f7b6d2"), ("#7f7f7f", "#c7c7c7"),
        ("#bcbd22", "#dbdb8d"),("#17becf", "#9edae5")
    ]

    def __init__(self, engine: MinkowskiEngine):
        self.engine = engine
        self.frames: List[ReferenceFrame] = []
        self.active_index: int = 0
        self.add_frame(v=0.0)

    def set_focus(self, index: int):
        for f in self.frames:
            f.is_active = (f.index == index)
        self.active_index = index

    def add_frame(self, v:float) -> ReferenceFrame:
        used_indices = {f.index for f in self.frames}
        free_index = next((i for i in range(11) if i not in used_indices), None)
        if free_index is None:
            raise RuntimeError
        if free_index == 0:
            new_frame = ReferenceFrame(self.engine, 0.0, 0, ("#404040", "#404040"))
        else:
            colors = self.tab20[free_index - 1]
            new_frame = ReferenceFrame(self.engine, v, free_index, colors)

        self.frames.append(new_frame)
        self.frames.sort(key=lambda x: x.index)
        self.set_focus(free_index)
        return new_frame

    def remove_frame(self):
        if self.active_index == 0:
            return
        deleted_index = self.active_index
        self.frames = [f for f in self.frames if f.index != self.active_index]
        self.set_focus(0)
        return deleted_index