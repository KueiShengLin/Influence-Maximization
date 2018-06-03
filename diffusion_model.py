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
        ith_activate_nodes = copy.deepcopy(s)

        for activated_node in ith_activate_nodes:
            for neighbor in self.g.neighbors(activated_node):
                w = len([node for node in self.g.neighbors(neighbor) if node in ith_activate_nodes])

                if neighbor not in ith_activate_nodes:
                    if random.random() <= 1 - (1 - self.p) ** w:
                        ith_activate_nodes.append(neighbor)

        return ith_activate_nodes

    def influence_spread(self):
        pass


if __name__ == '__main__':
    graph_data = pd.read_csv("graph30.txt", sep=" ", header=None)
    graph_data.columns = ['node_1', 'node_2']
    G = nx.from_pandas_edgelist(graph_data, target='node_1', source='node_2')

    IC = IndependentCascade(G, 0.1)

    ith_activate_nodes = IC.run([0, 1, 2, 3])

#
