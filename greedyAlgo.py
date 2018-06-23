from diffusion_model import IndependentCascade
from diffusion_model import LinearThreshold
import pandas as pd
import networkx as nx
import copy
import time


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

    start = time.time()
    print("iteration start!")
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

        print((i+1), "iteration time:", time.time() - start)

    return seed_set


def ic_celf(g, k=5, p=0.01, R=20):
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
    now_seed_influence = 0

    start = time.time()

    print("iteration start!")
    sv = {}
    for v in g.nodes():
        if v not in seed_set:
            sv[v] = 0
            union = copy.copy(seed_set)
            union.append(v)  # union = {seed_set U v}

            for round_time in range(R):
                sv[v] += len(IC.run(union))

            sv[v] = (sv[v] / R) - now_seed_influence

    influence = sorted(sv.items(), key=lambda x: x[1], reverse=True)  # x = sv, x[0] = sv.keys(), x[1] = sv.values()
    seed_set.append(influence[0][0])  # seed_set U (node which have highest score)
    now_seed_influence += influence.pop(0)[1]  # now seed influence = max infuence
    print(1, "iteration time:", time.time() - start)

    for i in range(1, k):
        sv = {}  # the score for node not in seed_set

        now_best = influence[0][0]

        union = copy.copy(seed_set)
        union.append(now_best)

        now_best_influence = 0
        for round_time in range(R):
            now_best_influence += len(IC.run(union))

        print(now_best_influence, now_seed_influence)
        now_best_influence = (now_best_influence / R) - now_seed_influence
        print(now_best_influence, influence[1][1])

        if now_best_influence >= influence[1][1]:
            seed_set.append(now_best)  # seed_set U (now_best)
            now_seed_influence += now_best_influence  # now seed influence = max infuence
            influence.pop(0)
            print((i + 1), "iteration time:", time.time() - start)

        else:
            for v in g.nodes():
                if v not in seed_set:
                    sv[v] = 0
                    union = copy.copy(seed_set)
                    union.append(v)  # union = {seed_set U v}

                    for round_time in range(R):
                        sv[v] += len(IC.run(union))

                    sv[v] = (sv[v] / R) - now_seed_influence

            influence = sorted(sv.items(), key=lambda x: x[1], reverse=True)  # x = sv, x[0] = sv.keys(), x[1] = sv.values()
            seed_set.append(influence[0][0])  # seed_set U (node which have highest score)
            now_seed_influence += influence.pop(0)[1]  # now seed influence = max infuence
            print((i+1), "iteration time:", time.time() - start)

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
    graph_data = pd.read_csv("data/NetHEPT.txt", sep=" ", header=None)
    graph_data.columns = ['node_1', 'node_2']
    G = nx.from_pandas_edgelist(graph_data, target='node_1', source='node_2')

    seed_set = ic_celf(G, k=30, p=.01, R=5)

    with open("seedset/ICGeneralGreedy.txt", 'w') as f:
        for seed in seed_set:
            f.write(str(seed) + "\n")

    IC = IndependentCascade(G, .01)
    seed_set = pd.read_csv("seedset/ICGeneralGreedy.txt", sep=" ", header=None)
    t = 0

    for i in range(5):
        t += len(IC.run(list(seed_set[0].values)))

    t = t / 20
    print(t)
#
