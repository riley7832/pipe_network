"""Reader for the steady-state subset of the EPANET INP format."""

from pathlib import Path

from .network import PipeNetwork
from .pipe import Fitting, Fluid, Node, Pipe, Pump, Reservoir


_FLOW_TO_M3S = {
    "CFS": 0.028316846592,
    "GPM": 0.0000630901964,
    "MGD": 0.0438126364,
    "IMGD": 0.0526167824,
    "AFD": 0.0142764102,
    "LPS": 0.001,
    "LPM": 1 / 60_000,
    "MLD": 1000 / 86_400,
    "CMH": 1 / 3600,
    "CMD": 1 / 86_400,
}


def read_inp(path: str | Path, fluid: Fluid | None = None) -> PipeNetwork:
    """Parse an EPANET INP file into a :class:`PipeNetwork`.

    Supports fixed-demand junctions, fixed-head reservoirs, open Darcy-Weisbach
    pipes (including pipe minor losses as explicit ``Fitting`` links), and pumps
    with a ``HEAD`` curve. Geometry, flow, and roughness values are converted to
    SI units. Demand patterns, tanks, valves, controls, closed/check-valve pipes,
    and non-Darcy-Weisbach headloss formulas are rejected because the current
    steady-state network model cannot represent them faithfully.

    Pump curve points are represented by piecewise-linear interpolation in head
    versus flow. The optional ``fluid`` overrides density and viscosity from the
    file's ``SPECIFIC GRAVITY`` and ``VISCOSITY`` options.
    """
    sections: dict[str, list[tuple[int, list[str]]]] = {}
    current_section: str | None = None
    for line_number, raw_line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        line = raw_line.split(";", 1)[0].strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            current_section = line[1:-1].strip().upper()
            sections.setdefault(current_section, [])
            continue
        if current_section is not None:
            sections.setdefault(current_section, []).append((line_number, line.split()))

    def error(line_number: int, message: str) -> ValueError:
        return ValueError(f"{path}:{line_number}: {message}")

    unsupported_sections = {
        "TANKS", "VALVES", "DEMANDS", "CONTROLS", "RULES", "EMITTERS", "STATUS"
    }
    unsupported = sorted(section for section in unsupported_sections if sections.get(section))
    if unsupported:
        raise ValueError(
            f"Unsupported INP sections for this steady-state parser: {', '.join(unsupported)}"
        )

    options: dict[str, list[str]] = {}
    for _, fields in sections.get("OPTIONS", []):
        if len(fields) >= 2 and fields[0].upper() == "SPECIFIC" and fields[1].upper() == "GRAVITY":
            options["SPECIFIC GRAVITY"] = fields[2:]
        elif len(fields) >= 2:
            options[fields[0].upper()] = fields[1:]
    flow_units = options.get("UNITS", ["GPM"])[0].upper()
    if flow_units not in _FLOW_TO_M3S:
        raise ValueError(f"Unsupported EPANET flow units: {flow_units}")
    flow_factor = _FLOW_TO_M3S[flow_units]
    length_factor = 0.3048 if flow_units in {"CFS", "GPM", "MGD", "IMGD", "AFD"} else 1.0
    diameter_factor = 0.0254 if length_factor == 0.3048 else 0.001
    roughness_factor = 0.0003048 if length_factor == 0.3048 else 0.001

    headloss = options.get("HEADLOSS", ["H-W"])[0].upper().replace(" ", "")
    if headloss not in {"D-W", "DW", "DARCY-WEISBACH"}:
        raise ValueError(
            "Only Darcy-Weisbach INP files are supported; set [OPTIONS] HEADLOSS D-W"
        )

    if fluid is None:
        specific_gravity = float(options.get("SPECIFIC GRAVITY", ["1.0"])[0])
        relative_viscosity = float(options.get("VISCOSITY", ["1.0"])[0])
        density = 1000.0 * specific_gravity
        fluid = Fluid(density=density, viscosity=0.001 * specific_gravity * relative_viscosity)

    network = PipeNetwork(fluid=fluid)
    nodes: dict[str, Node] = {}
    for section, rows in (("JUNCTIONS", sections.get("JUNCTIONS", [])), ("RESERVOIRS", sections.get("RESERVOIRS", []))):
        for line_number, fields in rows:
            if len(fields) < 2:
                raise error(line_number, f"Malformed [{section}] row")
            identifier = fields[0]
            if identifier in nodes:
                raise error(line_number, f"Duplicate node ID {identifier!r}")
            if section == "JUNCTIONS":
                if len(fields) > 3:
                    raise error(line_number, "Demand patterns are not supported")
                elevation = float(fields[1]) * length_factor
                demand = float(fields[2]) * flow_factor if len(fields) >= 3 else 0.0
                node = Node(identifier=identifier, demand=demand, elevation=elevation)
            else:
                if len(fields) > 3:
                    raise error(line_number, "Reservoir head patterns are not supported")
                head = float(fields[1]) * length_factor
                # EPANET reservoir input is fixed hydraulic head; represent it as
                # zero gauge pressure at an elevation equal to that head.
                node = Reservoir(identifier=identifier, demand=0.0, pressure=0.0, elevation=head)
            nodes[identifier] = node
            network.network.add_node(node)

    curves: dict[str, list[tuple[int, float, float]]] = {}
    for line_number, fields in sections.get("CURVES", []):
        if len(fields) < 3:
            raise error(line_number, "Malformed [CURVES] row; expected curve ID, x, y")
        curve_id = fields[0]
        curves.setdefault(curve_id, []).append(
            (line_number, float(fields[1]), float(fields[2]))
        )

    for line_number, fields in sections.get("PIPES", []):
        if len(fields) < 6:
            raise error(line_number, "Malformed [PIPES] row")
        identifier, start_id, end_id = fields[:3]
        try:
            start_node, end_node = nodes[start_id], nodes[end_id]
        except KeyError as exc:
            raise error(line_number, f"Pipe references unknown node {exc.args[0]!r}") from exc
        length = float(fields[3]) * length_factor
        diameter = float(fields[4]) * diameter_factor
        roughness = float(fields[5]) * roughness_factor
        minor_loss = 0.0
        status = "OPEN"
        if len(fields) >= 7:
            if fields[6].upper() in {"OPEN", "CLOSED", "CV"}:
                status = fields[6].upper()
            else:
                minor_loss = float(fields[6])
        if len(fields) >= 8:
            status = fields[7].upper()
        if status != "OPEN":
            raise error(line_number, f"Pipe status {status!r} is unsupported; only OPEN is supported")

        if minor_loss > 0:
            # Preserve the separate fitting abstraction by splitting the pipe
            # around an intermediate node and inserting a Fitting link.
            fitting_node = Node(
                identifier=("__inp_fitting_node__", identifier),
                demand=0.0,
                elevation=(start_node.elevation + end_node.elevation) / 2,
            )
            upstream = Pipe(
                identifier=f"{identifier}__upstream", start_node=start_node,
                end_node=fitting_node, length=length / 2, inner_diameter=diameter,
                absolute_roughness=roughness, flowrate=0.0,
            )
            # A fitting is represented as a zero-length link between two distinct
            # nodes; use a second coincident node so a DiGraph can store the link.
            fitting_end = Node(
                identifier=("__inp_fitting_end__", identifier),
                demand=0.0,
                elevation=fitting_node.elevation,
            )
            fitting = Fitting(
                identifier=f"{identifier}__minor_loss", start_node=fitting_node,
                end_node=fitting_end, loss_coefficient=minor_loss,
                reference_diameter=diameter,
            )
            downstream = Pipe(
                identifier=f"{identifier}__downstream", start_node=fitting_end,
                end_node=end_node, length=length / 2, inner_diameter=diameter,
                absolute_roughness=roughness, flowrate=0.0,
            )
            for item in (upstream, fitting, downstream):
                network.add_item(item)
        else:
            network.add_item(
                Pipe(
                    identifier=identifier, start_node=start_node, end_node=end_node,
                    length=length, inner_diameter=diameter,
                    absolute_roughness=roughness, flowrate=0.0,
                )
            )

    for line_number, fields in sections.get("PUMPS", []):
        if len(fields) < 5:
            raise error(line_number, "Malformed [PUMPS] row")
        identifier, start_id, end_id = fields[:3]
        try:
            start_node, end_node = nodes[start_id], nodes[end_id]
        except KeyError as exc:
            raise error(line_number, f"Pump references unknown node {exc.args[0]!r}") from exc
        properties = {fields[i].upper(): fields[i + 1] for i in range(3, len(fields) - 1, 2)}
        if "HEAD" not in properties:
            raise error(line_number, "Only pumps specified with a HEAD curve are supported")
        if "PATTERN" in properties or float(properties.get("SPEED", "1")) != 1.0:
            raise error(line_number, "Time-varying or non-unit pump speeds are unsupported")
        curve_id = properties["HEAD"]
        if curve_id not in curves or len(curves[curve_id]) < 2:
            raise error(line_number, f"Pump curve {curve_id!r} needs at least two points")
        pump_curve = tuple(
            (flow * flow_factor, head * length_factor)
            for _, flow, head in sorted(curves[curve_id], key=lambda point: point[1])
        )
        pump = Pump(
            identifier=identifier,
            start_node=start_node,
            end_node=end_node,
            shutoff_head=pump_curve[0][1],
            head_curve=pump_curve,
        )
        network.add_item(pump)

    if not network.items:
        raise ValueError("INP file contains no supported pipes or pumps")
    return network
