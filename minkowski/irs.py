import numpy as np
import matplotlib.pyplot as plt 
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import string

"""
Core module for Inertial Reference Systems (IRS). 
For simplicity, all velocities are in natural units (c=1).
Provides basic relativity functions and IRS Class.

These functions are intended for implemententation in the main code as the mathematical 
and physical foundation for plotting Minkowski 1+1D diagrams.
"""

# ----------- basic functions ----------- 

def gamma(v: float):
        """Lorentz factor for speed v (|v| < 1)."""
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

# ----------- classes ----------- 

@dataclass
class Event:
        """
        Class for spacetime events, with an associated color and invariant interval that 
        will help to define the type of event.
        """
        t: float
        x: float
        name: str
        color: str = ""

@dataclass
class InertialFrame:
        """
        1+1D inertial reference frame class including speed v and defined events.
        """
        name: str
        v: float
        color: str
        index: int = 0
        events: Dict[str, Event] = field(default_factory=dict)
        
        def add_event(self, t: float, x: float, frames: List['InertialFrame'], sr_rest):
                """
                When an event is added to a frame, an inverse Lorentz transformation is 
                automatically applied to it and is defined as an event in the rest frame.
                The event is then transformed to every moving frame and added to their 
                respective event dictionary. Every moving frame and its elements are 
                assigned a sufix depending on their index in the "frames" list.

                ex: An event "A2" is defined in the moving frame "S2". The event is then 
                defined in the rest frame (S) as "A", then transformed to the moving frame
                "S1" and saved as "A1".

                The invariant interval is also defined in every frame for further studies.
                """
                t_rest, x_rest = (inverse_lorentz_transform(t, x, self.v) 
                                  if self != sr_rest else (t, x))

                existing_names = [name.rstrip("0123456789") for name in sr_rest.events.keys()]
                base_name = generate_event_name(existing_names)
                sr_rest.events[base_name] = Event(t_rest, x_rest, base_name, sr_rest.color)

                for frame in frames:
                        t_new, x_new = lorentz_transform(t_rest, x_rest, frame.v)
                        event_label = base_name if frame == sr_rest else f"{base_name}{frame.index}"
                        frame.events[event_label] = Event(t_new, x_new, event_label, frame.color)

# dictionary for all the defined IRS
frames: List[InertialFrame] = []
# IRS at rest
sr_rest = InertialFrame("S", 0.0, "#333333")
frames.append(sr_rest)

def assign_color(frames, sri_index):
        """
        Add a new 1+1D reference frame to the frames list with a specific speed v and
        a color from a matplotlib colormap. Every frame is consistent in terms of sufix
        (assigned by its position in the list, being frames[0] the rest frame) and color
        with its defined elements.
        """
        if sri_index == 0:
                return "#333333"
        else:
                cmap = plt.get_cmap("tab10")
                color_index = (sri_index - 1) % 20
                return cmap(color_index)

def generate_event_name(existing_names: list) -> str:
        """
        Generate an event not present in the existing names list.
        """
        letters = string.ascii_uppercase
        n = 1
        
        while True:
                for i in range(26**n):
                        name = ""
                        num = i
                        for _ in range(n):
                                name = letters[num % 26] + name
                                num //= 26
                        if name not in existing_names:
                                return name
                n += 1