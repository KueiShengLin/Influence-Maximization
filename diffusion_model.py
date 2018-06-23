import networkx as nx
import pandas as pd
import random
import copy


class IndependentCascade:
    def __init__(self, g, p=.01):
        """
        init IC model
        :param g: networkx graph object
        :param p: independent probability
        """
        self.g = g
        self.p = p

    def run(self, s):
        """
        RanCas(S), run random cascade process
        :param s: ith activated nodes
        :return: (i+1)th activated nodes
        """
        activated_nodes = copy.deepcopy(s)

        for activated_node in activated_nodes:
            for neighbor in self.g.neighbors(activated_node):
                w = len([node for node in self.g.neighbors(neighbor) if node in activated_nodes])

                if neighbor not in activated_nodes:
                    if random.random() <= 1 - (1 - self.p) ** w:
                        activated_nodes.append(neighbor)

        return activated_nodes


class LinearThreshold:
    def __init__(self, g, weight="uniform"):
        """
        init LT model
        :param g: direct graph
        :param weight: choose weight calculate algorithm
        """

        assert type(g) == nx.DiGraph, "the input graph should be a nx.Digraph"

        self.g = g

        if weight == "uniform":
            self.weight = self.__uniform_weight()
        elif weight == "random":
            self.weight = self.__random_weight()

    def __uniform_weight(self):
        """
        Every edge euv have uniform weight (u,v)'s parallel edge / u's degree
        if graph not multigraph the weight is 1 / u's degree
        :return: weight list
        """
        weight_list = {}

        for node in self.g.nodes():
            u_in_edges = self.g.in_edges(node)

            for edge in u_in_edges:
                if edge not in weight_list.keys():
                    weight_list[edge] = 1 / len(u_in_edges)
                else:
                    weight_list[edge] += 1 / len(u_in_edges)

        return weight_list

    def __random_weight(self):
        """
        Every edge has random weight
        normalize each node in_edge so they weight have total sum = 1
        :return: weight list
        """
        weight_list = {}
        for edge in self.g.edges():
            weight_list[edge] = random.random()

        for node in self.g.nodes():
            u_in_edges = self.g.in_edges(node)

            total_random_weight = sum([weight_list[in_edge] for in_edge in u_in_edges])

            for in_edge in u_in_edges:
                weight_list[in_edge] = weight_list[in_edge] / total_random_weight

        return weight_list

    def run(self, s):
        """
        run LT model
        :param s: ith seed set
        :return: (i+1)th seed set
        """
        activated_nodes = copy.deepcopy(s)
        activate_threshold = {}
        now_activate_rate = {}

        for node in self.g.nodes():
            activate_threshold[node] = random.random()
            now_activate_rate[node] = 0

        for activated_node in activated_nodes:
            for neighbor in self.g.neighbors(activated_node):
                if neighbor not in activated_nodes:
                    now_activate_rate[neighbor] += self.weight[(activated_node, neighbor)]
                    if now_activate_rate[neighbor] >= activate_threshold[neighbor]:
                        activated_nodes.append(neighbor)

        return activated_nodes


if __name__ == '__main__':
    graph_data = pd.read_csv("graph30.txt", sep=" ", header=None)
    graph_data.columns = ['node_1', 'node_2']
    G = nx.from_pandas_edgelist(graph_data, source='node_1', target='node_2', create_using=nx.DiGraph())

    LT = LinearThreshold(G, "uniform")
    print(LT.run([2]))
#
