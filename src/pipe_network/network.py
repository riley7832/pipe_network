import networkx as nx

from .pipe import Fluid, Node, Pipe


class PipeNetwork:
    def __init__(self, fluid: Fluid):
        self.network = (
            nx.DiGraph()
        )  # undirected as we dont know the direction of flows to begin with
        self.fluid = fluid

    def add_pipe(self, pipe: Pipe):
        """Adds a pipe into the network
        Stores internally in nx.DiGraph()

        Args:
            pipe (Pipe): _description_
        """
        self.network.add_edge(pipe.start_node, pipe.end_node, pipe=pipe)

    def get_pipe_between(self, start_node: Node, end_node: Node) -> Pipe:
        return self.network.edges[(start_node, end_node)]["pipe"]

    @property
    def loops(self) -> list[list[Node]]:
        """Returns undirected loops

        Returns:
            list[list[Node]]: List of undirected loops defined by the nodes in each list
        """
        return nx.cycle_basis(self.network.to_undirected())

    @property
    def pipes(self) -> list[Pipe]:
        return [data["pipe"] for _, _, data in self.network.edges(data=True)]

    def initialise_flows(self) -> None:
        """Initialise pipe flows by splitting through a spanning tree"""

        # start with an undirected graph
        # get the pipes and the inlet nodes
        graph = self.network.to_undirected()
        inlets = [node for node in graph if node.demand < 0]
        if not inlets:
            raise ValueError("No inlet nodes")

        # Constructs a spanning tree
        tree = nx.bfs_tree(graph, source=inlets[0])
        # initialise the pipe flows to 0
        for pipe in self.pipes:
            pipe.flowrate = 0

        # Send flows outwards from the inlet
        # first sort the tree
        for node in nx.topological_sort(tree):
            # Get all the children of the current node
            children_nodes = list(tree.successors(node))
            # Checks for leaf node
            if not children_nodes:
                continue

            # Calculate the flow incoming to the node
            incoming_flow = 0
            for pipe in self.pipes:
                if pipe.end_node == node or pipe.start_node == node:
                    orientation = 1 if pipe.end_node == node else -1
                    incoming_flow += orientation * pipe.flowrate
            available_flow = incoming_flow - node.demand
            # This will split the flow evenly to the children
            downstream_share = available_flow / len(children_nodes)
            for child in children_nodes:
                pipe_to_child = self.get_pipe_between(start_node=node, end_node=child)
                pipe_to_child.flowrate = (
                    downstream_share
                    if pipe_to_child.start_node == node
                    else -downstream_share
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
