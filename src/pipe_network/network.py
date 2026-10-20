import networkx as nx
import numpy as np

from .pipe import Fluid, Node, Pipe


class PipeNetwork:
    def __init__(self, fluid: Fluid):
        self.network = (
            nx.DiGraph()
        )  # undirected as we dont know the direction of flows to begin with
        self.fluid = fluid

    def add_pipe(self, pipe: Pipe):
        self.network.add_edge(pipe.start_node, pipe.end_node, pipe=pipe)

    @property
    def loops(self) -> list[list[Node]]:
        """Returns undirected loops

        Returns:
            list[list[Node]]: List of undirected loops
        """
        return nx.cycle_basis(self.network.to_undirected())

    def initialise_flows(self) -> None:
        """Initialise pipe flows by minimising sum(r * Q**2).

        Assumptions:
        - Each node has a `demand` attribute (positive = withdrawal,
        negative = supply).
        - Each edge stores a Pipe object under "pipe".
        - Each Pipe has `start_node`, `end_node`, `flow`, and `r`.
        - Each pipe's `r` is positive.
        """

        edges = list(self.network.edges(data=True))
        nodes = list(
            dict.fromkeys(
                node
                for _, _, data in edges
                for node in (
                    data["pipe"].start_node,
                    data["pipe"].end_node,
                )
            )
        )

        n, m = len(nodes), len(edges)
        node_idx = {node: i for i, node in enumerate(nodes)}

        n, m = len(nodes), len(edges)

        A = np.zeros((n, m))
        r = np.zeros(m)
        b = np.array([node.demand for node in nodes], dtype=float)

        for j, (u, v, data) in enumerate(edges):
            pipe: Pipe = data["pipe"]

            A[node_idx[pipe.start_node], j] = 1.0
            A[node_idx[pipe.end_node], j] = -1.0
            r[j] = pipe.resistance_factor(self.fluid)

        if not np.isclose(b.sum(), 0.0, atol=1e-9):
            raise ValueError("Total demand and supply must balance.")

        # Remove one redundant continuity equation.
        A_reduced = A[:-1, :]
        b_reduced = -b[:-1]

        # Minimise sum(r * Q**2), subject to A @ Q = -b.
        W_inv = np.diag(1.0 / r)
        M = A_reduced @ W_inv @ A_reduced.T

        Q = W_inv @ A_reduced.T @ np.linalg.solve(M, b_reduced)

        for j, (_, _, data) in enumerate(edges):
            data["pipe"].flowrate = Q[j]


def solve_hardy_cross(
    network: PipeNetwork, tolerance: float = 1e-8, max_iterations=1000
):
    network.initialise_flows()
    for _ in range(max_iterations):
        max_correction = 0
        for loop in network.loops:
            # Shifts nodes one to the left, and puts the first one at the end
            # Zipping with the orgininal loop gives the start and end nodes for edges
            pairs = list(zip(loop, loop[1:] + loop[:1]))
            numerator = 0
            denominator = 0
            for u, v in pairs:
                if network.network.has_edge(u=u, v=v):
                    # Clockwise edge
                    edge = (u, v)
                    sign = 1
                elif network.network.has_edge(u=v, v=u):
                    # Counterclockwise edge
                    edge = (v, u)
                    sign = -1
                else:
                    raise ValueError(f"No edge connecting {v} and {u}")

                pipe: Pipe = network.network.edges[edge]["pipe"]
                flow = pipe.flowrate
                resistance_factor = pipe.resistance_factor(network.fluid)
                numerator += sign * resistance_factor * flow * abs(flow)
                denominator += 2.0 * resistance_factor * abs(flow)
            if denominator == 0:
                continue
            d_flow = -numerator / denominator
            max_correction = max(abs(d_flow), max_correction)
            for u, v in pairs:
                if network.network.has_edge(u, v):
                    edge, sign = (u, v), 1.0
                else:
                    edge, sign = (v, u), -1.0

                network.network.edges[edge]["pipe"].flowrate += sign * d_flow

        if max_correction < tolerance:
            return network

    raise RuntimeError("Hardy Cross did not converge")
