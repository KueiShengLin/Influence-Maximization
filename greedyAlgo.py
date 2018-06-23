from diffusion_model import IndependentCascade
from diffusion_model import LinearThreshold
import pandas as pd
import networkx as nx
import copy


def ic_general_greedy(g, k=5, p=0.1, R=20):
    """
    Find seed set S, use Wei Chen et al. algo1
    :param g: networkx graph object
    :param k: number of seed in S
    :param p: propagation probability
    :param R: number of rounds of simulations
    :return: Seed set
    """
    IC = IndependentCascade(g, p)
    seed_set = []

    for i in range(k):
        sv = {}  # the score for node not in seed_set

        for v in g.nodes():
            if v not in seed_set:
                sv[v] = 0
                union = copy.copy(seed_set)
                union.append(v)  # union = {seed_set U v}

                for round_time in range(R):
                    sv[v] += len(IC.run(union))

                sv[v] = sv[v] / R

        sorted_sv = sorted(sv.items(), key=lambda x: x[1], reverse=True)  # x = sv, x[0] = sv.keys(), x[1] = sv.values()
        seed_set.append(sorted_sv[0][0])   # seed_set U (node which have highest score)

    return seed_set


def lt_greedy(g, k=5, R=20):
    """
    Find seed set using LT model
    :param g: networkx direct graph object
    :param k: number of seed in S
    :return: seed set
    """
    assert type(g), "the input graph should be a nx.Digraph"

    LT = LinearThreshold(G)
    seed_set = []

    for i in range(k):
        sv = {}  # the score for node not in seed_set

        for v in g.nodes():
            if v not in seed_set:
                sv[v] = 0
                union = copy.copy(seed_set)
                union.append(v)  # union = {seed_set U v}

                for round_time in range(R):
                    sv[v] += len(LT.run(union))

                sv[v] = sv[v] / R

            sorted_sv = sorted(sv.items(), key=lambda x: x[1],
                               reverse=True)  # x = sv, x[0] = sv.keys(), x[1] = sv.values()
            seed_set.append(sorted_sv[0][0])  # seed_set U (node which have highest score)

        return seed_set


if __name__ == '__main__':
    graph_data = pd.read_csv("graph30.txt", sep=" ", header=None)
    graph_data.columns = ['node_1', 'node_2']
    G = nx.from_pandas_edgelist(graph_data, target='node_1', source='node_2', create_using=nx.DiGraph())

    LT = LinearThreshold(G, "uniform")

    seed_set = lt_greedy(G)
    t = 0
    for i in range(1):
        t += len(LT.run(seed_set))
    t = t
    # seed_set = general_greedy(G, 1, 0.i1, R=2000)
    print(t)

#
