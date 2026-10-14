import random
import numpy as np
from typing import List, Tuple
from itertools import tee

from .base import Node, Graph
from gym_PBN.utils.converters import logic_funcs_to_PBN_data
from gym_PBN.utils.logic.eval import LogicExpressionEvaluator


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


class PBNNode(Node):
    def __init__(self, index, data, genes):
        input_mask, truth_table, node, _ = data
        super(PBNNode, self).__init__(index, -1, node, node, False)

        self.input_mask = input_mask
        self.truth_table = truth_table

        predictor_genes = list(self.input_mask.nonzero()[0])

        # Test that the order of predictor genes is correct, i.e. the list of indexes of the
        # predictor genes should be stricly increasing to assure that they match the ordering
        # of predictor genes used to construct the truth table.
        assert all(x < y for x, y in pairwise(predictor_genes)), \
               'The order of predictor genes does not match the order used to construct the truth table!'

        self.predictors = [predictor_genes]
        self.used_by_agent = False


    def step(self, state, verbose=False):
        # relevant_nodes = tuple(state[i] for i in self.predictors[0])
        # proba = self.truth_table[relevant_nodes]
        # # this requires some finese in case of probabilistic boolean models
        # # but works just fine for non-probabilistic ones
        # self.value = int(proba)
        # return self.value

        # More efficient implementation
        state = np.array(state)
        # Current implementation is valid for BNs only, so len(self.predictors) == 1
        relevant_nodes = state[self.predictors[0]]
        # This requires some finese in case of probabilistic Boolean models
        # but works just fine for non-probabilistic ones
        self.value = int(self.truth_table[tuple(relevant_nodes)])

        return self.value        


class PBNGraph(Graph):
    def __init__(self, genes, logic_functions):

        # The implementation is valid for BNs only. PBNs are not implemented yet.
        try:
            assert all([len(funs) == 1 for funs in logic_functions])
        except AssertionError:
            raise NotImplementedError
        
        super(PBNGraph, self).__init__(2)
        self.genes = genes
        # pbn_data is a list of tuples (input_mask, truth_table, node, control), where 'input_mask'
        # is a Boolean vector indicating which nodes are parent nodes of 'node' and 'control' is
        # a Boolean argument of value sum(input_mask) == 0.
        pbn_data = logic_funcs_to_PBN_data(self.genes, logic_functions)

        # pbn_data contains data for individual nodes in the order of self.genes
        for gene, (_, _, node, _) in zip(self.genes, pbn_data):
            assert gene == node, 'Failed to properly initialise the BN Graph object: inconsistent ordering of nodes!'

        for i, data in enumerate(pbn_data):
            print(f'  spawning node {i}')
            self.nodes.append(PBNNode(i, data, self.genes))


    def step(self, changed_nodes: list = None, i=None):
        oldState = self.getState()
        i = random.randint(0, len(self.nodes) - 1) if i is None else i
        # while i in changed_nodes:
        #     i = random.randint(0, len(self.nodes) - 1)
        self.nodes[i].step(oldState)

        return self.getState()


    def get_adj_list(self):
        top_nodes = []
        bot_nodes = []

        for top_node in self.nodes:
            done = set()
            # Add self-loops
            top_nodes.append(top_node.index)
            bot_nodes.append(top_node.index)
            # Make sure that self-loops are added only once.
            done.add(top_node.index)

            for bot_node_id in top_node.predictors[0]:
                if bot_node_id not in done:
                    done.add(bot_node_id)
                    top_nodes.append(top_node.index)
                    bot_nodes.append(bot_node_id)

        return [top_nodes, bot_nodes]
