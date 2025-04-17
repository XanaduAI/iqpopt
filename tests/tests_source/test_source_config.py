# import jax.numpy as jnp
# import iqpopt as iqp
# import pennylane as qml
# from itertools import product
from pathlib import Path
# import numpy as np

n_qubits = 2
arr_gates_fn = ["local_gates", "multi_gens"]
arr_bitflip = [False, True]
arr_sparse = [False, True]
arr_spin_sym = [False, True]
arr_init_gates_coefs = [False, True]
seed = None # Values reused for each run if seed set instead of None

sigma = 0.1
n_ops = 10
n_samples = 10
max_batch_ops = 10_000
max_batch_samples = 100_000

parent_folder = Path(__file__).resolve().parent
results_filename = parent_folder / "test_source_results.csv"

# class TestCircuit(iqp.IqpSimulator):
#     def __init__(self,
#                  n_qubits: int,
#                  gates: list,
#                  device: str = "lightning.qubit",
#                  spin_sym: bool = False,
#                  init_gates: list = None,
#                  sparse: bool = False,
#                  bitflip: bool = False
#                 ):
#         super().__init__(n_qubits,
#                          gates,
#                          device,
#                          spin_sym,
#                          init_gates,
#                          sparse,
#                          bitflip
#                         )
    
#     #######################
#     #  Pennylane methods  #
#     #######################
    
#     def penn_obs(self, op: np.ndarray) -> qml.operation.Observable:
#         """Returns a pennylane observable from a bitstring representation.

#         Args:
#             op (np.ndarray): Bitstring representation of the Z operator.

#         Returns:
#             qml.Observable: Pennylane observable.
#         """
#         for i, z in enumerate(op):
#             if i==0:
#                 if z:
#                     obs = qml.Z(i)
#                 else:
#                     obs = qml.I(i)
#             else:
#                 if z:
#                     obs @= qml.Z(i)
#         return obs

#     def penn_op_expval_circ(self, params: np.ndarray, op: np.ndarray, init_coefs: list = None) -> qml.measurements.ExpectationMP:
#         """Defines the circuit that calculates the expectation value of the operator with the IQP circuit with pennylane tools.

#         Args:
#             params (np.ndarray): The parameters of the IQP gates.
#             op (np.ndarray): Bitstring representation of the Z operator.
#             init_coefs (list[float], optional): List or array of length len(init_gates) that specifies the fixed parameter
#                 values of init_gates.

#         Returns:
#             qml.measurements.ExpectationMP: Pennylane circuit with an expectation value.
#         """
#         self.iqp_circuit(params, init_coefs)
#         obs = self.penn_obs(op)
#         return qml.expval(obs)
    
#     def penn_op_expval(self, params: np.ndarray, op: np.ndarray, init_coefs: list = None) -> float:
#         """Calculates the expectation value of the operator with the IQP circuit with pennylane tools.

#         Args:
#             params (np.ndarray): The parameters of the IQP gates.
#             op (np.ndarray): Bitstring representation of the Z operator.
#             init_coefs (list[float], optional): List or array of length len(init_gates) that specifies the fixed parameter
#                 values of init_gates.

#         Returns:
#             float: Expectation value.
#         """
#         dev = qml.device(self.device, wires=self.n_qubits)
#         penn_op_expval_exe = qml.QNode(self.penn_op_expval_circ, dev)
#         return penn_op_expval_exe(params, op, init_coefs)

#     def penn_x_circuit(self, x: np.ndarray):
#         """Applies a pennylane circuit in order to initialize the qubits in the state of the bitstring x.

#         Args:
#             x (np.ndarray): Bitstring.
#         """
#         for i, b in enumerate(x):
#             if b:
#                 qml.X(i)

#     def penn_train_expval(self, x: np.ndarray, op: np.ndarray) -> qml.measurements.ExpectationMP:
#         """Defines the circuit that calculates the expectation value of the operator with the training circuit with pennylane tools.

#         Args:
#             x (np.ndarray): Bitstring representing the initialization of the circuit.
#             op (np.ndarray): Bitstring representing the observable that we want to measure.

#         Returns:
#             qml.measurements.ExpectationMP: Pennylane circuit with an expectation value.
#         """
#         self.penn_x_circuit(x)
#         obs = self.penn_obs(op)
#         return qml.expval(obs)
    
#     def penn_train_expval_dev(self, training_set: np.ndarray, op: np.ndarray) -> float:
#         """Calculates the expectation value of the operator with the training circuit with pennylane tools.

#         Args:
#             training_set (np.ndarray): The training set of samples from which we are trying to learn the distribution.
#             op (np.ndarray): The operator that is being measured.

#         Returns:
#             float: The result of the expectation value.
#         """
#         dev = qml.device(self.device, wires=self.n_qubits)
#         penn_train_expval = qml.QNode(self.penn_train_expval, dev)
#         tr_train = 0
#         for x in training_set:
#             tr_train += penn_train_expval(x, op)
#         tr_train /= len(training_set)
#         return tr_train
    
#     def penn_mmd_loss(self, params: np.ndarray, training_set: np.ndarray, sigma: float, init_coefs: list = None) -> float:
#         """Calculates the exact MMD Loss for the given IQP circuit with Pennylane tools.

#         Args:
#             params (np.ndarray): The parameters of the IQP gates.
#             training_set (np.ndarray): The training set of samples from which we are trying to learn the distribution.
#             sigma (float): Sigma parameter, the width of the kernel.
#             init_coefs (list[float], optional): List or array of length len(init_gates) that specifies the fixed parameter
#                 values of init_gates.

#         Returns:
#             float: The value of the loss.
#         """
#         loss = 0
#         p_MMD = (1-jnp.exp(-1/2/sigma))/2
#         for op in product([0,1], repeat=self.n_qubits):
#             op = jnp.array(op)
#             tr_iqp = self.penn_op_expval(params, op, init_coefs)
#             tr_train = self.penn_train_expval_dev(training_set, op)
#             loss += (1-p_MMD)**(self.n_qubits-op.sum()) * p_MMD**op.sum() * (tr_iqp*tr_iqp - 2*tr_iqp*tr_train + tr_train*tr_train)
#         return loss
