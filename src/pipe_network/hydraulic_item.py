from abc import ABC, abstractmethod
from collections.abc import Hashable
from dataclasses import dataclass
from itertools import pairwise

import numpy as np

from .fluids import Fluid, swamee_jain_friction_factor

gravity = 9.81


@dataclass
class Node:
    identifier: Hashable
    demand: float  # flow demand. Positive means the node consumes flow
    pressure: float | None = None  # fixed gauge pressure, Pa
    elevation: float = 0.0  # m above the chosen datum
    solved_pressure: float | None = None

    def __hash__(self) -> int:
        return hash(self.identifier)

    def __str__(self) -> str:
        return str(self.identifier)

    def __repr__(self) -> str:
        return str(self.identifier)


@dataclass
class Reservoir(Node):
    """Fixed-pressure boundary, normally specified by its water-surface pressure."""

    demand: float = 0.0
    pressure: float | None = 0.0


class HydraulicItem(ABC):
    """A directed connection between two nodes with a flow-dependent head change."""

    identifier: Hashable
    start_node: Node
    end_node: Node
    flowrate: float

    @abstractmethod
    def head_change(self, flowrate: float, fluid: Fluid) -> float:
        """Return head drop from start to end (negative means head is added)."""


@dataclass
class Pipe(HydraulicItem):
    identifier: Hashable
    start_node: Node
    end_node: Node
    length: float
    inner_diameter: float
    absolute_roughness: float
    flowrate: float

    @property
    def cross_sectional_area(self) -> float:
        return np.pi * (self.inner_diameter / 2) ** 2

    def resistance_factor(self, fluid):
        """This assumes circular pipe. It is calculated by rearranging the darcy weisbach equation
        for the flowrate - i.e. dP = f * (L/D) * (rho*u**2)/2
        Substitute u for the Q/A
        Express A in terms of diameter

        Args:
            fluid (_type_): _description_

        Returns:
            _type_: _description_
        """
        return (
            8
            * self.friction_factor(fluid=fluid)
            * self.length
            / (gravity * np.pi**2 * self.inner_diameter**5)
        )

    def friction_factor(self, fluid):
        return swamee_jain_friction_factor(pipe=self, fluid=fluid)

    def head_change(self, flowrate: float, fluid: Fluid) -> float:
        """Return signed Darcy-Weisbach head loss in the pipe's orientation."""
        self.flowrate = flowrate
        return self.resistance_factor(fluid) * flowrate * abs(flowrate)


@dataclass
class Pump(HydraulicItem):
    """Pump link with head curve H(Q) = shutoff_head - curve_coefficient * Q²."""

    identifier: Hashable
    start_node: Node
    end_node: Node
    shutoff_head: float  # m
    curve_coefficient: float = 0.0  # m / (m³/s)²
    flowrate: float = 0.0
    head_curve: tuple[tuple[float, float], ...] | None = None

    def head_gain(self) -> float:
        """Evaluate the pump curve at its current flow rate, in metres of head."""
        if self.head_curve:
            # Interpolate between curve points and linearly extrapolate beyond the ends.
            points = self.head_curve
            for (q0, h0), (q1, h1) in pairwise(
                points
            ):  # Gets 2 adjacent (flow, head) points on the pump curve
                if q0 <= self.flowrate <= q1:
                    return h0 + (h1 - h0) * (self.flowrate - q0) / (q1 - q0)
            (q0, h0), (q1, h1) = (
                (points[0], points[1])
                if self.flowrate < points[0][0]
                else (points[-2], points[-1])
            )
            return h0 + (h1 - h0) * (self.flowrate - q0) / (q1 - q0)
        return self.shutoff_head - self.curve_coefficient * self.flowrate**2

    def head_change(self, flowrate: float, fluid: Fluid) -> float:
        """Return negative head loss because a pump adds energy to the flow."""
        self.flowrate = flowrate
        return -self.head_gain()


@dataclass
class Fitting(HydraulicItem):
    """A local-loss element, such as a bend, valve, or contraction."""

    identifier: Hashable
    start_node: Node
    end_node: Node
    loss_coefficient: float
    reference_diameter: float
    flowrate: float = 0.0

    def head_change(self, flowrate: float, fluid: Fluid) -> float:
        """Return signed local head loss using the fitting's reference diameter."""
        self.flowrate = flowrate
        area = np.pi * self.reference_diameter**2 / 4
        velocity = flowrate / area
        return self.loss_coefficient * velocity * abs(velocity) / (2 * gravity)
