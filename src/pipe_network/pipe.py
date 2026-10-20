import math
from collections.abc import Hashable
from dataclasses import dataclass

import numpy as np

gravity = 9.81


@dataclass
class Fluid:
    density: float
    viscosity: float


def swamee_jain_friction_factor(pipe: Pipe, fluid: Fluid):
    fluid_velocity = abs(pipe.flowrate) / pipe.cross_sectional_area
    reynolds = fluid.density * fluid_velocity * pipe.inner_diameter / fluid.viscosity

    f_laminar = 64 / reynolds
    f_turbulent = (
        0.25
        / math.log10(
            pipe.absolute_roughness / (3.7 * pipe.inner_diameter) + 5.74 / reynolds**0.9
        )
        ** 2
    )

    if reynolds < 2300:
        return f_laminar

    elif reynolds > 5000:
        return f_turbulent

    weight = (reynolds - 2300) / (5000 - 2300)
    return (1 - weight) * f_laminar + weight * f_turbulent


@dataclass
class Node:
    identifier: Hashable
    demand: float  # flow demand. Positive means the node consumes flow

    def __hash__(self) -> int:
        return hash(self.identifier)

    def __str__(self) -> str:
        return str(self.identifier)

    def __repr__(self) -> str:
        return str(self.identifier)


@dataclass
class Pipe:
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
        return (
            8
            * self.friction_factor(fluid=fluid)
            * self.length
            / (gravity * np.pi**2 * self.inner_diameter**5)
        )

    def friction_factor(self, fluid):
        return swamee_jain_friction_factor(pipe=self, fluid=fluid)
