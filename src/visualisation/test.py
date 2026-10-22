from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import sympy as sp

from pipe_network.network import PipeNetwork, solve_network, solve_hardy_cross
from pipe_network.pipe import Fluid, Node, Pipe


def plot_digraph(network: PipeNetwork, figsize=(10, 7), seed=42):
    pos = nx.spring_layout(network.network, seed=seed)

    nx.draw_networkx(
        network.network,
        pos,
        with_labels=True,
        node_color="skyblue",
        node_size=1500,
        font_size=10,
        arrows=True,
        arrowsize=20,
        edge_color="gray",
    )

    plt.axis("off")
    plt.tight_layout()
    plt.show()


def compute_sources_and_sinks(G, tolerance=1e-9):
    net_flows = {node: 0.0 for node in G.nodes}

    for u, v, data in G.edges(data=True):
        q = data["item"].flowrate

        # Positive flow is u -> v
        net_flows[u] += q
        net_flows[v] -= q

    sources = {node: q for node, q in net_flows.items() if q > tolerance}

    sinks = {node: -q for node, q in net_flows.items() if q < -tolerance}

    return sources, sinks


def visualise_network(network, file_name: str):
    G = network.network
    sources, sinks = compute_sources_and_sinks(G)

    pos = nx.spring_layout(G, seed=42, k=1.5, iterations=100, scale=2)

    # Draw ordinary junctions
    ordinary_nodes = [n for n in G.nodes if n not in sources and n not in sinks]
    nx.draw_networkx_nodes(
        G, pos, nodelist=ordinary_nodes, node_size=600, node_color="lightblue"
    )

    # Draw source nodes (green squares)
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=list(sources),
        node_size=700,
        node_color="lightgreen",
        node_shape="s",
        label="Source",
    )

    # Draw sink nodes (red triangles)
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=list(sinks),
        node_size=700,
        node_color="salmon",
        node_shape="v",
        label="Sink",
    )

    # Node labels
    nx.draw_networkx_labels(G, pos, font_size=10)

    # Source and sink rate labels
    node_rate_labels = {
        **{n: f"Source: {q:.4f}" for n, q in sources.items()},
        **{n: f"Sink: {q:.4f}" for n, q in sinks.items()},
    }

    # Offset rate labels slightly from node positions
    rate_pos = {n: (x, y - 0.15) for n, (x, y) in pos.items()}
    nx.draw_networkx_labels(
        G,
        rate_pos,
        labels=node_rate_labels,
        font_size=8,
        font_color="black",
    )

    # Draw edges in the direction of flow
    for u, v, data in G.edges(data=True):
        flow: float = data["item"].flowrate
        edge = (u, v) if flow >= 0 else (v, u)

        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=[edge],
            arrows=True,
            arrowstyle="->",
            arrowsize=20,
            connectionstyle="arc3,rad=0.05",
        )

    # Pipe flow labels
    edge_labels = {
        (u, v): f"Q={abs(data['item'].flowrate):.4f}"
        for u, v, data in G.edges(data=True)
    }
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    plt.title("Hardy Cross Pipe Network")
    plt.legend()
    plt.axis("off")
    plt.tight_layout()
    # Resolve output from the repository root, regardless of the working directory.
    output_dir = Path(__file__).resolve().parents[0] / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / f"{file_name}.png", bbox_inches="tight")
    plt.show()


def main():
    water = Fluid(density=1000, viscosity=0.001)
    start_node = Node("start", demand=-1)
    end_node = Node("end", demand=1, pressure=0)
    n1 = Node("n1", demand=0)
    n2 = Node("n2", demand=5)
    n3 = Node("n3", demand=-5)
    pipe1 = Pipe(
        "pipe1",
        start_node=start_node,
        end_node=n1,
        length=2,
        inner_diameter=0.1,
        absolute_roughness=0.0005,
        flowrate=1,
    )
    pipe2 = Pipe(
        "pipe2",
        start_node=start_node,
        end_node=n3,
        length=100,
        inner_diameter=0.2,
        absolute_roughness=0.0005,
        flowrate=1,
    )
    pipe3 = Pipe(
        "pipe3",
        start_node=n1,
        end_node=n2,
        length=5,
        inner_diameter=0.25,
        absolute_roughness=0.0005,
        flowrate=1,
    )
    pipe4 = Pipe(
        "pipe4",
        start_node=n2,
        end_node=end_node,
        length=10,
        inner_diameter=0.2,
        absolute_roughness=0.0005,
        flowrate=1,
    )
    pipe5 = Pipe(
        "pipe5",
        start_node=n2,
        end_node=n3,
        length=10,
        inner_diameter=0.2,
        absolute_roughness=0.0005,
        flowrate=1,
    )
    pipe6 = Pipe(
        "pipe6",
        start_node=n1,
        end_node=end_node,
        length=200,
        inner_diameter=0.15,
        absolute_roughness=0.0005,
        flowrate=1,
    )
    pipe7 = Pipe(
        "pipe7",
        start_node=n3,
        end_node=end_node,
        length=250,
        inner_diameter=0.155,
        absolute_roughness=0.0005,
        flowrate=1,
    )
    network = PipeNetwork(fluid=water)
    network.add_pipe(pipe1)
    network.add_pipe(pipe2)
    network.add_pipe(pipe3)
    network.add_pipe(pipe4)
    network.add_pipe(pipe5)
    network.add_pipe(pipe6)
    network.add_pipe(pipe7)
    # plot_digraph(network)
    # print(network.loops)
    # solve_network(network, max_iterations=1000)
    # visualise_network(network=network, file_name="New Solver")

    solve_hardy_cross(network, max_iterations=1000)
    visualise_network(network=network, file_name="Hardy Cross")
    for pipe in network.pipes:
        print(pipe.flowrate)


def symp():
    f, l, d, rho, u, q = sp.symbols("f l d rho u q")
    eqn = f * (l / d) * rho * u**2 / 2

    flow_eqn = sp.Eq(q, u * (sp.pi * d**2) / 4)
    u_expr = sp.solve(flow_eqn, u)[0]
    print(u_expr)
    res = eqn.subs({u: u_expr})
    print(res)


if __name__ == "__main__":
    main()
    # symp()
