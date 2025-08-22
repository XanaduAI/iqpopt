# IQPopt - IQP circuit optimization with JAX
IQPopt is a package designed for fast optimization of parameterized Instantaneous quantum polynomial (IQP) circuits 
using JAX. Research paper avaliable [here](https://arxiv.org/abs/2501.04776). 

## Installation

Install with

```python
pip install .
```

or in editable mode with 
```python
pip install -e .
```

## Creating a circuit

The package can be used to optimize *parameterized IQP circuits*. These are circuits comprised of gates 
$\text{exp}(i\theta_j X_j)$, where the generator $X_j$ is a tensor product of Pauli X operators acting on some subset of qubits 
and $\theta_j$ is a trainable parameter. Input states and measurements are diagonal in the computational (Z) basis.

To define such a circuit (with input state $\vert 0 \rangle$) we need to specify the number of qubits and the parameterized gates 

```python
import iqpopt as iqp
from iqpopt.utils import local_gates

n_qubits = 2
gates = local_gates(n_qubits, 2) 

circuit = iqp.IqpSimulator(n_qubits, gates)
```

Each element of `gates` corresponds to a trainable parameter, and is given by a list of lists
that specifies the generators of the parameter. 

For example, in the above the function
`local_gates` returns `gates = [[[0]],[[1]],[[0,1]]]` which specifies three trainable parameters with gate generators
$X_0$, $X_1$ and $X_0X_1$. 

> **Note**: For very large problems it can be useful to initialize the circuit with the option `sparse=True`. 
> This uses scipy sparse matrix multiplication in place of JAX and can be significantly more memory efficient.

## Expectation values
IQPopt has been designed for fast evaluation of expectation values of Pauli Z tensors.

To estimate the expectation value of a Pauli Z tensor, we represent the operator as a binary string. The estimation
uses a Monte Carlo method whose precision is controlled by `n_samples`. 

```python
import jax
import jax.numpy as jnp

op = jnp.array([0, 1]) #binary array representing Z_1
params = jnp.ones(len(circuit.gates))
n_samples = 1000
key = jax.random.PRNGKey(42)

expval, std = circuit.op_expval(params, op, n_samples, key)
```
returns an estimate of $\langle Z_1 \rangle$ as well as the standard deviation of the estimator.

The package also allows for fast batch evaluation of expectation values. If we specify a batch of Z 
operators by an array

```python
ops = jnp.array([[0,1],[1,0],[1,1]]) #Z_1, Z_0, Z0Z1
```
we can also batch evaluate the expectation values in parallel:
```python
expvals, stds = circuit.op_expval(params, ops, n_samples, key)
```

> **Note**: The estimation of each expectation value in the batch is unbiased, however the estimators may be correlated.
> This effect can be reduced by increasing n_samples in order to reduce the variance of each estimator, or by 
> using the option `indep_estimates=True to return uncorrelated estimates (at the cost of longer runtime).


## Training

We can train our circuit with built-in methods. We first define a loss function
```python
def loss(params, circuit, ops, n_samples, key):
    expvals = circuit.op_expval(params, ops, n_samples, key)[0]
    return jnp.sum(expvals)
```

The first argument must be named `params` and corresponds to the trainable
parameters. We can then train the circuit as follows

```python
import numpy as np
import matplotlib.pyplot as plt

optimizer = "Adam" 
stepsize = 0.001
n_iters = 1000
params_init = np.random.normal(0, 1/np.sqrt(n_qubits), len(circuit.gates))
ops = np.array([[1,1], [1,0], [0,1]])
n_samples = 1000

loss_kwargs = {
    "params": params_init,
    "circuit": circuit,
    "ops": ops,
    "n_samples": n_samples,
}

trainer = iqp.Trainer(optimizer, loss, stepsize)
trainer.train(n_iters, loss_kwargs)

params = trainer.final_params
plt.plot(trainer.losses)
```

Automatic stopping of training is possible using the `convergence_interval` option of `train`; see the docstring for more info. 

## Stochastic bitflip model
One can replace the quantum circuit by an analogous bitflipping model described in arxiv:2501.04776 by initializing the circuit 
with the `bitflip=True` option:

```python
circuit = iqp.IqpSimulator(n_qubits, gates, bitflip=True)
```
This can be useful to judge if the IQP model is making use of interference. Since the bitflipping model is classical, one can also 
sample from this model for large values of `n_qubits`.

# Generative machine learning with IQP circuits
## Training for generative machine learning tasks
We can also view the circuit as a generative model and train it using the maximum mean discrepancy (MMD) distance as a loss function.
```python
import iqpopt.gen_qml as gen
from iqpopt.gen_qml.utils import median_heuristic

n_qubits = 10

#toy dataset of low weight bitstrings
X_train = np.random.binomial(1,0.5, size=(1000, n_qubits))
X_train = X_train[np.where(X_train.sum(axis=1)<5)]

gates = local_gates(n_qubits, 2)
circuit = iqp.IqpSimulator(n_qubits, gates)
params_init = np.random.normal(0, 1/np.sqrt(n_qubits), len(gates))

loss = gen.mmd_loss_iqp #MMD loss
sigma = median_heuristic(X_train) #bandwidth for MMD

loss_kwargs = {
    "params": params_init,
    "iqp_circuit": circuit,
    "ground_truth": X_train,
    "sigma": sigma,
    "n_ops": 1000,
    "n_samples": 1000,
}

trainer = iqp.Trainer("Adam", loss, stepsize=0.01)
trainer.train(n_iters=500, loss_kwargs=loss_kwargs)

params = trainer.final_params
plt.plot(trainer.losses)
```
The MMD loss is estimated using a Monte Carlo method; larger values of `n_ops` and `n_samples` result in more precise 
estimates. For small circuits, we can generate new samples
```python
samples = circuit.sample(params, shots=100)
```
For large circuits this is not tractable due to the complexity of sampling from IQP distributions. 


## Evaluating the generative model

To evaluate the model, we can use the MMD distance to a test set, or the Kernel Generalized Empirical Likelihood (KGEL);
see Suman Ravuri et al. in [Understanding Deep Generative Models with Generalized Empirical Likelihoods](https://arxiv.org/abs/2306.09780).

### Kernel Generalized Empirical Likelihood (KGEL)

```python
#test points from same distribution
X_test = np.random.binomial(1,0.5, size=(1000, n_qubits))
X_test = X_test[np.where(X_test.sum(axis=1)<5)]

n_witness = 10 
witness_points = X_test[-n_witness:] #witness points for KGEL
test_data = X_test[:-n_witness] #test data for KGEL

kgel, p_kgel = gen.kgel_opt_iqp(circuit, params, witness_points, test_data, 
                             sigma, n_ops=1000, n_samples=1000, key=jax.random.PRNGKey(42))
```

`p_kgel` is the optimal probability distribution returned from the convex optimization.
