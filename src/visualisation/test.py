import matplotlib.pyplot as plt
import networkx as nx

from pipe_network.network import PipeNetwork, solve_hardy_cross
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


def main():
    water = Fluid(density=1000, viscosity=0.001)
    start_node = Node("start", demand=-1)
    end_node = Node("end", demand=1)
    n1 = Node("n1", demand=0)
    n2 = Node("n2", demand=0)
    n3 = Node("n3", demand=0)
    pipe1 = Pipe(
        "pipe1",
        start_node=start_node,
        end_node=n1,
        length=10,
        inner_diameter=0.2,
        absolute_roughness=0.0005,
        flowrate=1,
    )
    pipe2 = Pipe(
        "pipe2",
        start_node=start_node,
        end_node=n3,
        length=10,
        inner_diameter=0.2,
        absolute_roughness=0.0005,
        flowrate=1,
    )
    pipe3 = Pipe(
        "pipe3",
        start_node=n1,
        end_node=n2,
        length=10,
        inner_diameter=0.2,
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
    network = PipeNetwork(fluid=water)
    network.add_pipe(pipe1)
    network.add_pipe(pipe2)
    network.add_pipe(pipe3)
    network.add_pipe(pipe4)
    network.add_pipe(pipe5)
    # plot_digraph(network)
    # print(network.loops)
    solve_hardy_cross(network, max_iterations=1000)
    loop = network.loops[0]
    print(loop)
    print(loop[1:], loop[:1])
    print(loop[1:] + loop[:1])


if __name__ == "__main__":
    main()
