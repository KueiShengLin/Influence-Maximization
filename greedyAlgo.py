from diffusion_model import IndependentCascade
import pandas as pd
import networkx as nx
import copy


def general_greedy(g, k=5, p=0.1, R=20):
    """
    Find seed set S, use Wei Chen et al. algo1
    :param g: networkx graph object
    :param k: number of seed in S
    :param p: propagation probability
    :param R: number of rounds of simulations
    :return: Seed set S
    """
    IC = IndependentCascade(g, p)
    S = []

    for i in range(k):
        sv = {}  # the score for node not in S

        for v in g.nodes():
            if v not in S:
                sv[v] = 0
                union = copy.copy(S)
                union.append(v)  # union = {S U v}

                for round_time in range(R):
                    sv[v] += len(IC.run(union))

                sv[v] = sv[v] / R

        sorted_sv = sorted(sv.items(), key=lambda x: x[1], reverse=True)  # x = sv, x[0] = sv.keys(), x[1] = sv.values()
        S.append(sorted_sv[0][0])   # S U (node which have highest score)

    return S


if __name__ == '__main__':
    graph_data = pd.read_csv("graph30.txt", sep=" ", header=None)
    graph_data.columns = ['node_1', 'node_2']
    G = nx.from_pandas_edgelist(graph_data, target='node_1', source='node_2')

    seed_set = general_greedy(G, 1, 0.1, R=2000)
    print(seed_set)

#
