import networkx as nx
import numpy as np

from .pipe import Fluid, HydraulicItem, Node, Pipe, gravity


class PipeNetwork:
    def __init__(self, fluid: Fluid):
        self.network = (
            nx.DiGraph()
        )  # undirected as we dont know the direction of flows to begin with
        self.fluid = fluid

    def add_pipe(self, pipe: Pipe):
        """Add a pipe using the general hydraulic-item registration method."""
        self.add_item(pipe)

    def add_item(self, item: HydraulicItem):
        """Add a directed hydraulic item between its start and end nodes.

        The item object is stored on the graph edge and supplies its own
        flow-dependent head change to the network solver.
        """
        self.network.add_edge(item.start_node, item.end_node, item=item)

    def get_pipe_between(self, start_node: Node, end_node: Node) -> HydraulicItem:
        """Return the item on the directed edge from ``start_node`` to ``end_node``.

        The legacy method name is kept for callers that only use pipes.
        """
        if self.network.has_edge(start_node, end_node):
            return self.network.edges[(start_node, end_node)]["item"]
        if self.network.has_edge(end_node, start_node):
            return self.network.edges[(end_node, start_node)]["item"]
        raise KeyError(f"No hydraulic item connects {start_node} and {end_node}")

    @property
    def loops(self) -> list[list[Node]]:
        """Returns undirected loops

        Returns:
            list[list[Node]]: List of undirected loops defined by the nodes in each list
        """
        return nx.cycle_basis(self.network.to_undirected())

    @property
    def items(self) -> list[HydraulicItem]:
        """List every hydraulic item registered as a network edge."""
        return [data["item"] for _, _, data in self.network.edges(data=True)]

    @property
    def pipes(self) -> list[Pipe]:
        """List only pipe items for code that needs pipe-specific properties."""
        return [item for item in self.items if isinstance(item, Pipe)]

    def initialise_flows(self) -> None:
        """Initialise pipe flows by splitting through a spanning tree"""

        # start with an undirected graph
        # get the pipes and the inlet nodes
        graph = self.network.to_undirected()
        inlets = [node for node in graph if node.demand < 0]
        if not inlets:
            raise ValueError("No inlet nodes")

        # Add a temporary root connected to every inlet so each source starts
        # with its own outgoing flow in the spanning tree.
        root = object()
        spanning_graph = graph.copy()
        spanning_graph.add_node(root)
        spanning_graph.add_edges_from((root, inlet) for inlet in inlets)
        tree = nx.bfs_tree(spanning_graph, source=root)
        # initialise the pipe flows to 0
        for pipe in self.items:
            pipe.flowrate = 0

        # Work from the leaves towards the inlets. Each tree branch carries
        # the sum of demands downstream of that branch, preserving continuity.
        subtree_demand: dict[Node | object, float] = {}
        for node in reversed(list(nx.topological_sort(tree))):
            if node is root:
                continue

            downstream_demand = node.demand + sum(
                subtree_demand[child] for child in tree.successors(node)
            )
            subtree_demand[node] = downstream_demand

            parent = next(tree.predecessors(node))
            if parent is root:
                # The synthetic root only connects the inlet trees; it is not
                # a real network node and therefore has no hydraulic link.
                continue

            item = self.get_pipe_between(start_node=parent, end_node=node)
            item.flowrate = (
                downstream_demand
                if item.start_node == parent
                else -downstream_demand
            )


def solve_hardy_cross(
    network: PipeNetwork, tolerance: float = 1e-8, max_iterations=1000
):
    network.initialise_flows()
    for _ in range(max_iterations):
        max_correction = 0
        for loop in network.loops:
            # We loop through all the loops in the network
            # the loop is a list of nodes defining a closed loop
            loop: list[Node]

            # Shifts nodes one to the left, and puts the first one at the end
            # Zipping with the orgininal loop gives the start and end nodes for edges
            node_pairs = list(zip(loop, loop[1:] + loop[:1]))
            numerator = 0
            denominator = 0
            for start_node, end_node in node_pairs:
                # Traverse around the loop
                # First grab the edge and the direction of the edge
                # The sign is indicating the direction of traversal around the loop
                # It does not indicate the direction of flow
                if network.network.has_edge(u=start_node, v=end_node):
                    # Clockwise edge
                    edge_node_tuple = (start_node, end_node)
                    sign = 1
                elif network.network.has_edge(u=end_node, v=start_node):
                    # Counterclockwise edge
                    edge_node_tuple = (end_node, start_node)
                    sign = -1
                else:
                    raise ValueError(f"No edge connecting {end_node} and {start_node}")

                # Gets the pipe out of the network
                pipe: Pipe = network.get_pipe_between(
                    start_node=edge_node_tuple[0], end_node=edge_node_tuple[1]
                )
                flow = pipe.flowrate

                # The resistance factor is the rearranged form of the darcy weisbach equation
                if not isinstance(pipe, Pipe):
                    raise TypeError(
                        "solve_hardy_cross supports pipe items only; use solve_network for other items"
                    )
                resistance_factor = pipe.resistance_factor(network.fluid)

                # We need to sum the head loss

                # This sums to find sum{rQ^{n}}
                # Idea is to find the total head loss in the loop
                # Having one signed `flow` keeps the flow direction intact
                # The sign keeps the loop traversal direction intact
                numerator += sign * resistance_factor * flow * abs(flow)

                # This is summing to find sum{nrQ^{n-1}} without reference to direction
                # ALl values are positive
                denominator += 2.0 * resistance_factor * abs(flow)

            if denominator == 0:
                continue

            # This is the flow correction
            # We need to apply this to each edge
            d_flow = -numerator / denominator
            max_correction = max(abs(d_flow), max_correction)

            # Traverse the nodes again
            for start_node, end_node in node_pairs:
                # Recall that sign preserves the direction around the loop we traverse
                if network.network.has_edge(start_node, end_node):
                    edge_node_tuple, sign = (start_node, end_node), 1.0
                else:
                    edge_node_tuple, sign = (end_node, start_node), -1.0

                # Apply the flow correction
                network.get_pipe_between(
                    start_node=edge_node_tuple[0], end_node=edge_node_tuple[1]
                ).flowrate += sign * d_flow

        if max_correction < tolerance:
            return network

    raise RuntimeError("Hardy Cross did not converge")


def solve_network(
    network: PipeNetwork, tolerance: float = 1e-9, max_iterations: int = 100
):
    """Solve flows and nodal pressures using continuity and energy equations.

    Nodes with ``pressure`` set are fixed-head boundaries. Pipe demand is positive
    for consumption. Pressures are gauge Pa and elevations are metres.
    """
    # Each unknown link flow and free-node head has one matching energy or
    # continuity equation, respectively.

    # Builds some useful data stuctures
    nodes = list(network.network.nodes)
    edges = list(network.network.edges(data=True))
    # Fixed pressure nodes - ie ones set by user
    fixed_pressure_nodes: list[Node] = [n for n in nodes if n.pressure is not None]
    if not fixed_pressure_nodes:
        raise ValueError("At least one node must have a fixed pressure")
    # Pressures not set by user
    free_pressure_nodes: list[Node] = [n for n in nodes if n.pressure is None]
    if not edges:
        raise ValueError("Network has no links")
    # Dictionary keyed on the defined pressure node and value being the set pressure
    fixed_nodal_pressures: dict[Node, float] = {
        n: n.elevation + n.pressure / (network.fluid.density * gravity)
        for n in fixed_pressure_nodes
        if n.pressure is not None
    }
    number_of_flows = len(edges)
    # Initialises a vector to store the guesses - this means residual calcs
    # can be done on a single vector
    # We first store flowrates [f1, f2, f3, ..., fn] followed by pressure values
    # final matrix looks like [f1, f2, ... fn, h1, h2, ..., hn]
    linkage_flowrate_guess_matrix = np.zeros(
        number_of_flows + len(free_pressure_nodes), dtype=float
    )
    # Loops through the edges and initialises flowrate guess to either a defined flow or a small flow
    for i, (_, _, data) in enumerate(edges):
        link = data["item"]
        linkage_flowrate_guess_matrix[i] = link.flowrate or 1e-4
    # Loops through the nodes and sets their heads
    for col_idx, node in enumerate(free_pressure_nodes):
        # Use elevation as the initial total-head guess (zero pressure head).
        linkage_flowrate_guess_matrix[number_of_flows + col_idx] = node.elevation

    def residual(values):
        """Gets the error/residual for the current guess

        Args:
            values (np.NDArray): takes the linkage_flowrate_guess_matrix

        Returns:
            np.NDArray: Residuals for each variable
        """
        flows = values[:number_of_flows]
        heads = dict(fixed_nodal_pressures)
        # updates the heads for free pressure nodes to the guess in values
        heads.update(
            {
                node: values[number_of_flows + j]
                for j, node in enumerate(free_pressure_nodes)
            }
        )
        out = []
        # Loop through all the nodes to compute flow continuity
        for free_node in free_pressure_nodes:
            net_flow_into_free_node = 0.0
            for i, (edge_start_node, edge_end_node, _) in enumerate(edges):
                # Positive link flow follows the stored start_node -> end_node direction.
                # We add flows if edge is directed into the node
                if edge_end_node == free_node:
                    net_flow_into_free_node += flows[i]
                # or subract the flow if directed away from the node
                if edge_start_node == free_node:
                    net_flow_into_free_node -= flows[i]

            # This is the difference between net flow and node demand
            # Essentially a flow residual on conservation of mass
            out.append(net_flow_into_free_node - free_node.demand)

        # now we can loop through the edges to compute nodal pressures
        for i, (edge_start_node, edge_end_node, data) in enumerate(edges):
            # Get the actual edge item
            link = data["item"]
            # This guesses flowrate
            flow = flows[i]
            # The head loss associated with this flow rate
            loss = link.head_change(flow, network.fluid)
            # The item's signed head change includes losses and gains in its orientation.
            # This calculates the difference between the guessed heads at either end of the edge
            # and then compares to the computed head loss through the hydraulic item
            # This acts as the residual between the head loss guess and the computed head loss
            out.append(heads[edge_start_node] - heads[edge_end_node] - loss)
        return np.asarray(out)

    for _ in range(max_iterations):
        # Estimate the Jacobian numerically, then take a least-squares Newton step.
        # We get the residuals of flows and head losses
        residuals = residual(linkage_flowrate_guess_matrix)

        # np.linalg.norm calculates the size of a vector
        # ord=np.inf means we take the largest value in the residuals vector
        # Checks if residual is small
        # this is no different to np.max(np.abs(residuals)) < tolerance
        # but allows for changing the calculation method through `ord`
        if np.linalg.norm(residuals, ord=np.inf) < tolerance:
            # If we enter this, we have converged

            # Loops through and gets the actual edge item
            for i, (_, _, data) in enumerate(edges):
                # Sets the flowrate to the guess
                data["item"].flowrate = float(linkage_flowrate_guess_matrix[i])
            for node in free_pressure_nodes:
                # Sets the pressure to the solved pressure
                # uses rho * g * h
                node.solved_pressure = (
                    (
                        linkage_flowrate_guess_matrix[
                            number_of_flows + free_pressure_nodes.index(node)
                        ]
                        - node.elevation
                    )
                    * network.fluid.density
                    * gravity
                )
                # return the network
            return network

        # Recall the jacobian is the gradient matrix
        # This is how the residual changes as the guess changes
        # rows correspond to the residual, columns are the flow/pressure guesses
        jacobian_matrix = np.empty((len(residuals), len(linkage_flowrate_guess_matrix)))

        # This is column wise - i.e loops through each variable its guessed
        for col_idx in range(len(linkage_flowrate_guess_matrix)):
            # Steps are small (1e-6 * the flow/head guess)
            # we will add this step to the current jacobian
            step = 1e-6 * max(1.0, abs(linkage_flowrate_guess_matrix[col_idx]))

            # Store a second copy of the guess matrix so we can compare
            # the two before and after stepping the given variable
            adjusted_guess_matrix = linkage_flowrate_guess_matrix.copy()
            adjusted_guess_matrix[col_idx] += step

            # Now we can calculate the derivative (the change) between before and after
            # We calculate the residual for every flow/head (row wise) hence the [:, j]
            jacobian_matrix[:, col_idx] = (
                residual(adjusted_guess_matrix) - residuals
            ) / step

        # This is an interesting step
        # linalg.lstsq provides an approximate solution `x` to the equation A @ x = b (where @ is matrix multiplication)
        # recall matrix multiplication is different to the cross product and the dot product
        #
        # If we set A equal to the derivative (jacobian), and b equal to the negative residuals
        # We should get an x vector that is very close to the negative residuals
        # By using the negative residuals, we can cancel out the actual calculated residual
        #
        # The solution is stored in delta
        delta, *_ = np.linalg.lstsq(jacobian_matrix, -residuals, rcond=None)
        if not np.all(np.isfinite(delta)):
            # If we get an infinite value we will hit an error
            break

        # Now we can just add the required change to the guesses
        linkage_flowrate_guess_matrix += delta

    raise RuntimeError("Hydraulic network solver did not converge")
