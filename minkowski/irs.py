import numpy as np
from typing import Tuple, List

class MinkowskiEngine:
    """
    Computational core for kinematics in special relativity (1+1)D. Manages 
    a centralized coordinate matrix to optimize Lorentz  transformations 
    via vectorized operations, implementing the fundamental Minkowski
    geometric structure defined by the metric tensor.
    """
    def __init__(self):
        """
        Initializes the Minkowski space with a (+, -) metric signature and
        establishes the standard covariant basis vectors in the rest frame.
        """
        self.rest = np.empty((2,0))
        self.metric = np.diag([1, -1])
        self.e0 = np.array([[1],[0]])
        self.e1 = np.array([[0],[1]])

    @property
    def light_vectors(self) -> np.ndarray:
        """
        Generates the basis vectors defining the boundary of the light cone,
        satisfying $s^2 = 0$.
        """
        return np.array([[1.0, 1.0], [1.0, -1.0]])

    def add_event(self, t: float, x: float) -> int:
        """
        Adds a spacetime event into the central coordinate matrix, returning
        the column index associated with the event.
        """
        index = self.rest.shape[1]
        self.rest = np.hstack([self.rest, np.array([[t], [x]])])
        return index
    
    def remove_event(self, index: int):
        """
        Deletes events from the coordinate matrix via NaN masking, preserving 
        the dimensional structure of the matrix and the column indices of all 
        subsequent events.
        """
        if 0 <= index < self.rest.shape[1]:
            self.rest[:, index] = np.nan
        else:
            raise IndexError(f"Event {index} out of bounds.")

    @staticmethod
    def lorentz_matrix(v: float) -> np.ndarray:
        """
        Defines the Lorentz boost matrix for a given relative velocity in
        natural units (c=1), enforcing the physical speed limit abs(v)<1 
        in the definition of the Lorentz factor.
        """
        if abs(v) >= 1: raise ValueError("Velocity must be strictly between -1 and 1.")
        gamma = 1 / np.sqrt(1 - v**2)
        return np.array([[gamma, -v*gamma], [-v*gamma, gamma]])

    def boost(self, x: np.ndarray, v: float) -> np.ndarray:
        """
        Applies the Lorentz transformation from the rest frame to project 
        coordinates into an inertial frame moving at velocity v.
        """
        if v == 0: return x
        return self.lorentz_matrix(v) @ x
    
    def causal_structure(self, x: np.ndarray, epsilon = 1e-9) -> Tuple[float, str]:
        """
        Evaluates the invariant interval to classify the geometric causality 
        of a four-vector. Handles floating-point imprecision via a defined
        epsilon tolerance for null intervals.
        """
        s2 = (x.T @ self.metric @ x).item()
        if s2 > epsilon: causality = "timelike"
        elif s2 < -epsilon: causality = "spacelike"
        else: causality = "lightlike"
        return (s2, causality)


class Event:
    """
    Abstraction of a singular point in spacetime, acting as a pointer to 
    the central coordinate matrix defined in the engine, calculating its
    position given a velocity for subsequent graphical representation.
    """
    def __init__(self, engine: MinkowskiEngine, index: int, frame_index: int = 0):
        self.engine = engine
        self.index = index
        self.frame_index = frame_index

    def coordinates(self, v: float = 0.0) -> np.ndarray:
        """
        Calculates the event's contravariant coordinates relative to a 
        moving observer.
        """
        x_rest = self.engine.rest[:, [self.index]]
        return self.engine.boost(x_rest, v)

    def causality(self) -> Tuple[float, str]:
        """
        Determines the invariant interval relative to the origin.
        """
        x = self.engine.rest[:, [self.index]]
        return self.engine.causal_structure(x)


class Segment:
    """
    Represents a discrete spacetime interval bounded by two events. Useful
    for evaluating kinematic phenomena such as time dilation and length
    contraction across the defined reference frames.
    """
    def __init__(self, engine: MinkowskiEngine, index1: int, index2: int, frame_index: int = 0):
        self.engine = engine
        self.index1 = index1
        self.index2 = index2
        self.frame_index = frame_index

    def coordinates(self, v: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes the boosted coordinates of both bounding events.
        """
        x1_rest = self.engine.rest[:, [self.index1]]
        x2_rest = self.engine.rest[:, [self.index2]]
        return (self.engine.boost(x1_rest, v), self.engine.boost(x2_rest, v))

    def coordinate_deltas(self, v: float = 0.0) -> Tuple[float, float]:
        """
        Calculates the contravariant vector components in the boosted frames.
        """
        p1, p2 = self.coordinates(v)
        dt_prime = p2[0, 0] - p1[0, 0]
        dx_prime = p2[1, 0] - p1[1, 0]
        return dt_prime, dx_prime

    def causality(self) -> Tuple[float, str]:
        """
        Evaluates the invariant interval separating the two bounding events.
        """
        dx = self.engine.rest[:, [self.index2]] - self.engine.rest[:, [self.index1]]
        return self.engine.causal_structure(dx)


class ReferenceFrame:
    """
    Defines the Inertial Reference System (IRS), maintaining specific properties 
    for every observer moving at constant velocity relative to the rest frame.
    """
    def __init__(self, engine: MinkowskiEngine, v: float, index: int, colors: Tuple[str, str]):
        self.engine = engine
        self.v = v
        self.index = index
        self.colors = colors
        self.is_active = False

    @property
    def color(self) -> str:
        return self.colors[0] if self.is_active else self.colors[1]
    
    def axes(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes the contravariant coordinates of the moving observer axes
        as projected onto the rest frame by the application of an inverse
        Lorentz transformation.
        """
        t_axis = self.engine.boost(self.engine.e0, -self.v)
        x_axis = self.engine.boost(self.engine.e1, -self.v)
        return t_axis, x_axis


class FrameManager:
    """
    Manages the inertial observers preserving the absolute rest frame,
    maintaining visual coherence and computational stability in the 
    rendering pipeline.
    """
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
        """
        Defines a new IRS with its respective color assignment.
        """
        used_indices = {f.index for f in self.frames}
        free_index = next((i for i in range(11) if i not in used_indices), None)
        if free_index is None:
            raise RuntimeError("Maximum number of reference frames reached.")
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
        """
        Purges the currently focused IRS, returning the deleted index and 
        preventing the removal of the fundamental rest frame at index 0.
        """
        if self.active_index == 0:
            return -1
        deleted_index = self.active_index
        self.frames = [f for f in self.frames if f.index != self.active_index]
        self.set_focus(0)
        return deleted_index