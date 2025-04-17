import sys
import os
import time
import jax
import numpy as np
from iqpopt.utils import local_gates
import iqpopt.gen_qml as genq
import pandas as pd
import iqpopt as iqp
import test_source_config as config

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import pennylane_functions as tests

for bitflip in config.arr_bitflip:
    for gates_fn in config.arr_gates_fn:
        for init_gates_coefs in config.arr_init_gates_coefs:
            for spin_sym in config.arr_spin_sym:
                for sparse in config.arr_sparse:
                    
                    beginning = time.time()
                    
                    np.random.seed(config.seed)
                    
                    print(
                        "gates_fn:", gates_fn,
                        "bitflip:", bitflip,
                        "sparse:", sparse,
                        "spin_sym:", spin_sym,
                        "init_gates_coefs:", init_gates_coefs
                    )

                    gates = local_gates(config.n_qubits, 1)
                    if gates_fn == "multi_gens":
                        gates = [[gates[0][0], gates[1][0]]] + gates[2:]
                        
                    if init_gates_coefs:
                        init_gates = gates.copy()
                        init_coefs = np.random.uniform(0, 2*np.pi, len(init_gates))
                    else:
                        init_gates = None
                        init_coefs = None

                    start = time.time()
                    circuit = iqp.IqpSimulator(
                        config.n_qubits,
                        gates,
                        spin_sym=spin_sym,
                        init_gates=init_gates,
                        sparse=sparse,
                        bitflip=bitflip,
                    )
                    print(f"Circuit creation: {time.time() - start:.4f} s")

                    params_init = np.random.uniform(0, 2*np.pi, len(gates))
                    op = np.random.randint(0, 2, (config.n_qubits, ))
                    key = jax.random.PRNGKey(np.random.randint(0, 99999))
                    
                    start = time.time()
                    expval, std = circuit.op_expval(
                        params_init,
                        op,
                        config.n_samples,
                        key,
                        init_coefs,
                        max_batch_samples=config.max_batch_samples,
                        max_batch_ops=config.max_batch_ops
                    )
                    print(f"Circuit op_expval: {time.time() - start:.4f} s")
                    
                    start = time.time()
                    expval_penn = tests.penn_op_expval(
                        circuit,
                        params_init,
                        op,
                        init_coefs
                    )
                    print(f"Pennylane op_expval: {time.time() - start:.4f} s")
                    
                    training_set = np.random.randint(0, 2, (config.n_samples, config.n_qubits))
                    
                    start = time.time()
                    loss = genq.mmd_loss_iqp(
                        params_init,
                        circuit,
                        training_set,
                        config.sigma,
                        config.n_ops,
                        config.n_samples,
                        key,
                        init_coefs,
                        max_batch_samples=config.max_batch_samples,
                        max_batch_ops=config.max_batch_ops
                    )
                    print(f"Circuit MMD loss: {time.time() - start:.4f} s")
                    
                    start = time.time()
                    loss_penn = tests.penn_mmd_loss(
                        circuit,
                        params_init,
                        training_set,
                        config.sigma,
                        init_coefs
                    )
                    print(f"Pennylane MMD loss: {time.time() - start:.4f} s")
                    
                    trainer = iqp.Trainer("Adam", genq.mmd_loss_iqp, stepsize=0.01)
                    loss_kwargs = {
                        "params": params_init,
                        "iqp_circuit": circuit,
                        "ground_truth": training_set,
                        "sigma": config.sigma,
                        "n_ops": config.n_ops,
                        "n_samples": config.n_samples,
                        "key": key,
                        "init_coefs": init_coefs,
                        "max_batch_samples": config.max_batch_samples,
                        "max_batch_ops": config.max_batch_ops,
                    }
                    
                    start = time.time()
                    trainer.train(5, loss_kwargs)
                    print(f"Training MMD loss: {time.time() - start:.4f} s")
                    
                    test_tbl = {
                        "n_qubits": [config.n_qubits],
                        "n_ops": [config.n_ops],
                        "n_samples": [config.n_samples],
                        "gates_fn": [gates_fn],
                        "bitflip": [bitflip],
                        "sparse": [sparse],
                        "spin_sym": [spin_sym],
                        "init_gates_coefs": [init_gates_coefs],
                        "std": [std],
                        "expval": [expval],
                        "expval_penn": [expval_penn],
                        "loss": [loss],
                        "loss_penn": [loss_penn],
                    }
                    
                    
                    if os.path.isfile(config.results_filename):
                        pd.DataFrame(test_tbl).to_csv(config.results_filename, index=False, mode='a', header=False)
                    else:
                        pd.DataFrame(test_tbl).to_csv(config.results_filename, index=False)
                    
                    print(f"Total time: {time.time() - beginning:.4f} s")