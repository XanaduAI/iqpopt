import itertools
from itertools import combinations

import networkx as nx
import jax.numpy as jnp
import numpy as np
from scipy.sparse import csr_matrix

##### GATE CONSTUCTOR FUNCTIONS #######

def local_gates(n_qubits: int, max_weight=2):
    """
    Generates a gate list for an IqpSimulator object containing all gates whose generators have Pauli weight
    less or equal than max_weight.
    :param n_qubits: The number of qubits in the gate list
    :param max_weight: maximum Pauli weight of gate generators
    :return (list[list[list[int]]]): gate list object for IqpSimulator
    """
    gates = []
    for weight in np.arange(1, max_weight+1):
        for gate in combinations(np.arange(n_qubits), weight):
            gates.append([list(gate)])
    return gates

def gate_lists_to_arrays(gate_lists: list, n_qubits: int) -> list:
    """Transforms the gates parameter into a list of arrays of 0s and 1s.

    Args:
        gate_lists (list[list[list[int]]]): Gates list for IqpSimulator object.
        n_qubits (int): number of qubits in the return arrays

    Returns:
        list: Gates parameter in list of arrays form.
    """

    gate_arrays = []
    for gates in gate_lists:
        arr = np.zeros([len(gates), n_qubits])
        for i, gate in enumerate(gates):
            for j in gate:
                arr[i, j] = 1.
        gate_arrays.append(jnp.array(arr))
    return gate_arrays


def nodes_within_distance(G, source, distance):
    """
    Given a graph G, returns all nodes on within a specified distance of a source node.
    :param G: A networkx graph object
    :param source: source node
    :param distance: maximum distance from source node
    :return: list of nodes within specified distance
    """

    current_level = {source}
    visited = {source}
    all_nodes_within_distance = set(current_level)

    for _ in range(distance):
        next_level = set()
        for node in current_level:
            for neighbor in G.neighbors(node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    next_level.add(neighbor)
        current_level = next_level
        all_nodes_within_distance.update(current_level)

    all_nodes_within_distance.discard(source)
    return list(all_nodes_within_distance)


def nearest_neighbour_gates(G, distance: int, max_weight=2):
    """
    For every node on a graph G, find all other nodes within a specified distance, and create all gates
    up to a maximum weight from these nodes. All such gates are returned as a gate list to be used
    with IQPSimulator.
    :param G: networkx graph that determines the distances, or a path reference to a networkx adjacency list file
    :param distance: maximum distance to consider
    :param max_weight: maxmium weight of the generators of the gates
    :return: list of lists specifying the gates
    """

    if isinstance(G, str):
        G = nx.read_adjlist(G, nodetype=int)

    gates = []
    for source in list(G.nodes):
        neighbours = nodes_within_distance(G, source, distance)
        for weight in range(0, max_weight):
            for combo in itertools.combinations(neighbours, weight):
                new_gate = list(combo)+[source]
                new_gate = [int(elem) for elem in new_gate]
                perms = list(itertools.permutations(new_gate))
                perms = [[list(perm)] for perm in perms]
                if not any([perm in gates for perm in perms]):
                    gates.append([new_gate])

    return gates


def nearest_neighbour_lattice_gates(height: int, width: int, distance: int, max_weight=2):
    """
    Implements nearest_neighbour_gates for the specific case where the graph is a 2D lattice
    with periodic boundary conditions
    :param height: lattice height
    :param width: lattice width
    :param distance: maximum distance
    :param max_weight: maximum weight
    :return: list of lists specifying the gates
    """
    G = nx.grid_2d_graph(height, width, periodic=True)
    # convert labels to integers
    mapping = {(i, j): i * width + j for i in range(height)
               for j in range(width)}
    G = nx.relabel_nodes(G, mapping)
    return nearest_neighbour_gates(G, distance, max_weight)


def expand_gate_list(gate_lists, max_idx, n_ancilla, max_weight=2):
    """
    Given a list of gates and a maximum index max_idx, construct a new gate list of max_idx+n_ancilla bits, where
    for each gate in gate_lists we add a new gate that includes the ancilla, as long as the weight is below max_weight.
    e.g. if [[0,1]] is in gate_lists and we add a single ancilla, the gate [[0,1,2]] will be added if max_weight>=3.
    We also add the single bit gates to the ancillas.
    :param gate_lists: input list of gates
    :param max_idx: the maximum index of the input list of gates (i.e. the number of bits they act on)
    :param n_ancilla: The number of ancillas to connect
    :param max_weight: The maximum weight of the new gates to be added
    :return: the new gate list
    """
    new_gate_lists = list(gate_lists)
    for gates in gate_lists:
        for gate in gates:
            if len(gate) < max_weight:
                for i in range(max_idx, max_idx+n_ancilla):
                    new_gate = gate + [i]
                    new_gate_lists.append([new_gate])
    for i in range(max_idx, max_idx + n_ancilla):
        new_gate_lists.append([[i]])
    return new_gate_lists


def gates_from_covariance(data, n_gates, return_local=False):
    """
    constructs the covariance matrix of the data and returns those pairs of gates that have the strongest
    correlations
    :param data: The dataset of shape (n_samples, n_features)
    :param n_gates: The number of gates to return
    :return: list of lists specifying gates
    """
    args_sorted = np.argsort(-np.abs(np.reshape(np.cov((2*data-1).T), -1)))
    d = data.shape[-1]
    best_gates = [[[i]] for i in range(d)] if return_local else []
    for arg in args_sorted:
        idx = np.unravel_index(arg, [d, d])
        if idx[0] != idx[1]:
            if idx not in best_gates and (idx[1], idx[0]) not in best_gates:
                best_gates.append([list(idx)])
        if len(best_gates) == n_gates + return_local*d:
            break
    return best_gates


def random_gates(n_gates, max_idx, min_weight=None, max_weight=None):
    """
    Returns a list of random gates between a specified min and max weight.
    The distribution of weights is uniform.
    :param n_gates: number of gates
    :param max_idx: maximum index appearing in any gate
    :param min_weight: minimum weight of a gate (number of ones)
    :param max_weight: maximum weighht of a gate
    :return:
    """
    if min_weight is None:
        min_weight = 1

    if max_weight is None:
        max_weight = max_idx
    gates = []
    for __ in range(n_gates):
        weight = int(np.random.choice(np.arange(min_weight, max_weight+1)))
        gate = list(np.random.choice(np.arange(max_idx), weight, replace=False))
        gates.append([gate])
    return gates

##### PARAMETER INITIALIZATIONS #######

def initialize_from_data(gates_list: list, data: jnp.array, scale=-1, param_noise=0.):
    """
    Given a dataset and a gate specification, returns initial parameters such that

    - for each gate of the form [[i,j]], the corresponding parameter is set to the covariance between the ith and jth
        dimension of the dataset (using pm1 data) times the scale factor.
    - for each gate of the form [[i]] the corresponding parameter is set to the mean value of the ith dimension of the
        dataset times pi/2
    - parameters of all other gates are set with a normal distribution with standard deviation param_noise

    if scale is set to -1, a specific default heuristic is used.

    :param gates_list (list[list[list]]]): gate lists specifying the gates of the cirucit
    :param data (array): dataset of binary features
    :return: The parameter array
    """
    cov_mat = np.cov((2*data-1).T)  # use pm1 data for covariance
    np.fill_diagonal(cov_mat, 0.)
    max_cov = np.max(np.abs(cov_mat))
    means = np.mean(data, axis=0)
    params = []

    if scale == -1:
        scale = np.sqrt(data.shape[-1]/np.sum(cov_mat**2)*2)*max_cov/np.pi

    for gate in gates_list:
        if len(gate) == 1:
            gen = gate[0]
            if len(gen) == 1:
                params.append(np.arcsin(np.sqrt(means[gen[0]])))
            elif len(gen) == 2:
                params.append(cov_mat[gen[0], gen[1]]*scale*np.pi/max_cov)
            else:
                params.append(np.random.normal(0, param_noise))
        else:
            params.append(np.random.normal(0, param_noise))
    return jnp.array(params)

##### CIRCUIT ANALYSIS FUNCTIONS #######

def construct_convariance_matrix(circuit, params, n_samples, key, init_coefs: list = None, indep_estimates=False,
                                 max_batch_ops=None, max_batch_samples=None):
    """
    Construct the covariance matrix of an IQP cicuit.
    Args:
        params (IqpSimulator): The IQP circuit given by the class IqpSimulator.
        params (jnp.ndarray): The parameters of the IQP gates.
        n_samples (int): Number of samples used to calculate the IQP expectation values.
        key (Array): Jax key to control the randomness of the process.
        init_coefs (list[float], optional): List or array of length len(init_gates) that specifies the fixed parameter
                values of init_gates.
        indep_estimates (bool): Whether to use independent estimates of the ops in a batch (takes longer). Defaults to False.
        max_batch_ops (int): Maximum number of operators in a batch of op_expval. Defaults to None.
        max_batch_samples (int): Maximum number of samples in a batch of op_expval. Defaults to None.

    Returns:
        Array: The covariance matrix
    """
    ops_lists = [[[i, j]]
                 for i in range(circuit.n_qubits) for j in range(i + 1)]
    ops = np.concatenate(gate_lists_to_arrays(ops_lists, circuit.n_qubits))
    expvals = circuit.op_expval(params, ops, n_samples, key, init_coefs, indep_estimates=indep_estimates,
                                max_batch_ops=max_batch_ops, max_batch_samples=max_batch_samples)[0]
    expval_mat = np.zeros((circuit.n_qubits, circuit.n_qubits))
    count = 0
    for i in range(circuit.n_qubits):
        for j in range(i + 1):
            expval_mat[i, j] = expvals[count]
            expval_mat[j, i] = expvals[count]
            count += 1
    cov_mat = np.zeros((circuit.n_qubits, circuit.n_qubits))
    for i in range(circuit.n_qubits):
        for j in range(i + 1):
            if i == j:
                cov_mat[i, j] = 1.0 - expval_mat[i, i] * expval_mat[j, j]
            else:
                cov_mat[i, j] = expval_mat[i, j] - \
                    expval_mat[i, i] * expval_mat[j, j]
                cov_mat[j, i] = cov_mat[i, j]

    return cov_mat
