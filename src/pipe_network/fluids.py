import math
from dataclasses import dataclass

from .hydraulic_item import Pipe


@dataclass
class Fluid:
    density: float
    viscosity: float


def swamee_jain_friction_factor(pipe: Pipe, fluid: Fluid):
    fluid_velocity = abs(pipe.flowrate) / pipe.cross_sectional_area
    reynolds = fluid.density * fluid_velocity * pipe.inner_diameter / fluid.viscosity
    if reynolds == 0:
        return 0
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
