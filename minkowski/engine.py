import numpy as np
from typing import Tuple, List, cast

class MinkowskiEngine:
    """
    Computational core for kinematics in special relativity (1+1)D. Manages 
    a centralized coordinate matrix to optimize Lorentz transformations 
    via vectorized operations, implementing the fundamental Minkowski
    geometric structure defined by the metric tensor.
    """
    def __init__(self) -> None:
        """
        Initializes the Minkowski space with a (+, -) metric signature and 
        establishes the standard covariant basis vectors in the rest frame.
        Defines an initial dynamic array for event management.
        """
        self._capacity = 16
        self._next_index = 0
        self._free_indices: set[int] = set()
        self.rest = np.full((2, self._capacity), np.nan)
        
        self.metric = np.diag([1, -1])
        self.e0 = np.array([[1],[0]])
        self.e1 = np.array([[0],[1]])
        self.light_cone = np.array([[1.0, 1.0], [1.0, -1.0]])

    def add_event(self, t: float, x: float) -> int:
        """
        Adds a spacetime event into the central coordinate matrix, returning
        the column index associated with the event. Implements capacity 
        doubling prioritizing the CPU cost.
        """
        if self._free_indices:
            index = self._free_indices.pop()
        else:
            if self._next_index >= self._capacity:
                self._capacity *= 2
                new_rest = np.full((2, self._capacity), np.nan)
                new_rest[:, :self._next_index] = self.rest
                self.rest = new_rest
            index = self._next_index
            self._next_index += 1
        self.rest[:, index] = [t, x]
        return index
    
    def remove_event(self, index: int) -> None:
        """
        Deletes events from the coordinate matrix via NaN masking, preserving 
        the dimensional structure of the matrix and the column indices of all 
        subsequent events.
        """
        if 0 <= index < self._next_index and index not in self._free_indices:
            self.rest[:, index] = np.nan
            self._free_indices.add(index)
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
        return cast(np.ndarray, self.lorentz_matrix(v) @ x)
    
    def active_coordinates(self, v: float = 0.0) -> np.ndarray:
        """
        Computes a new submatrix formed by all defined non-NaN events and boosts 
        them in a single vectorized operation.
        """
        sub_rest = ~np.isnan(self.rest[0, :self._next_index])
        active_rest = self.rest[:, :self._next_index][:, sub_rest]
        if active_rest.size == 0:
            return active_rest
        return self.boost(active_rest, v)
    
    def causal_structure(self, x: np.ndarray, epsilon: float = 1e-9) -> Tuple[float, str]:
        """
        Evaluates the invariant interval to classify the geometric causality 
        of a four-vector. Handles floating-point imprecision via a defined
        epsilon tolerance for null intervals.
        """
        s2 = float((x[0]**2-x[1]**2).item())
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
    def __init__(self, engine: MinkowskiEngine, index: int):
        self.engine = engine
        self.index = index

    def coordinates(self, v: float = 0.0) -> np.ndarray:
        """
        Calculates the event's contravariant coordinates relative to a 
        moving observer.
        """
        x_rest = self.engine.rest[:, self.index:self.index+1]
        return self.engine.boost(x_rest, v)

    def causality(self) -> Tuple[float, str]:
        """
        Determines the invariant interval relative to the origin.
        """
        x = self.engine.rest[:, self.index:self.index+1]
        return self.engine.causal_structure(x)


class Segment:
    """
    Represents a discrete spacetime interval bounded by two events. Useful
    for evaluating kinematic phenomena such as time dilation and length
    contraction across the defined reference frames.
    """
    def __init__(self, engine: MinkowskiEngine, index1: int, index2: int):
        self.engine = engine
        self.index1 = index1
        self.index2 = index2

    def coordinates(self, v: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes the boosted coordinates of both bounding events.
        """
        x1_rest = self.engine.rest[:, self.index1:self.index1+1]
        x2_rest = self.engine.rest[:, self.index2:self.index2+1]
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
        dx = self.engine.rest[:, self.index2:self.index2+1] - self.engine.rest[:, self.index1:self.index1+1]
        return self.engine.causal_structure(dx)


class ReferenceFrame:
    """
    Defines the Inertial Reference System (IRS), maintaining specific properties 
    for every observer moving at constant velocity relative to the rest frame.
    """
    def __init__(self, engine: MinkowskiEngine, v: float, index: int):
        self.engine = engine
        self.v = v
        self.index = index
    
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
    def __init__(self, engine: MinkowskiEngine):
        self.engine = engine
        self.frames: List[ReferenceFrame] = []
        self.add_frame(v=0.0)

    def add_frame(self, v:float) -> ReferenceFrame:
        """
        Defines a new IRS with a defined relative velocity.
        """
        used_indices = {f.index for f in self.frames}
        free_index = next((i for i in range(11) if i not in used_indices), None)
        if free_index is None:
            raise RuntimeError("Maximum number of reference frames reached.")
        new_frame = ReferenceFrame(self.engine, v, free_index)

        self.frames.append(new_frame)
        self.frames.sort(key=lambda x: x.index)
        return new_frame

    def remove_frame(self, index: int) -> None:
        """
        Purges a specific IRS by its index, preventing the removal of the 
        fundamental rest frame at index 0.
        """
        original_count = len(self.frames)
        if index == 0:
            raise ValueError("The fundamental rest frame cannot be removed.")
        self.frames = [f for f in self.frames if f.index != index]
        if len(self.frames) == original_count:
            raise IndexError("The reference frame does not exist.")