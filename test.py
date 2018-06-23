import unittest
import networkx as nx
import pandas as pd
from diffusion_model import LinearThreshold


class LTTest(unittest.TestCase):
    def setUp(self):
        graph_data = pd.read_csv("data/NetHEPT.txt", sep=" ", header=None)
        graph_data.columns = ['node_1', 'node_2']
        G = nx.from_pandas_edgelist(graph_data, source='node_1', target='node_2', create_using=nx.DiGraph())

        self.g = G
        self.LT = LinearThreshold(G)

    def test_random_weight_equal_one(self):
        weight_list = self.LT.random_weight()

        for node in self.g.nodes():
            u_in_edges = self.g.in_edges(node)

            real_total_random_weight = sum([weight_list[in_edge] for in_edge in u_in_edges])

            my_total_random_weight = 0
            for in_edge in u_in_edges:
                my_total_random_weight += weight_list[in_edge] / real_total_random_weight

            self.assertAlmostEquals(my_total_random_weight, real_total_random_weight)


if __name__ == '__main__':
    unittest.main()

#
