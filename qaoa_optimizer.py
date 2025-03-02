# File: examples/optimizers/qaoa/qaoa_optimizer.py

from qubots.base_optimizer import BaseOptimizer
import pennylane as qml
import numpy as np
from pennylane import numpy as pnp

class QAOAOptimizer(BaseOptimizer):
    """
    QAOA Optimizer for QUBO problems.
    This implementation uses the Ising mapping to build the cost unitary.
    
    For each QAOA layer:
      - The cost unitary U_C is implemented as:
          For each qubit i with local field h[i]: apply RZ(2 * gamma * h[i]).
          For each edge (i,j): apply:
              CNOT(i, j) → RZ(2 * gamma * J[(i,j)]) on qubit j → CNOT(i, j)
      - The mixer unitary U_B is implemented as applying RX(-2 * beta) on every qubit.
    """
    def __init__(self, num_layers=1, max_iters=100, learning_rate=0.5, n_shots=1000, verbose=False):
        self.num_layers = num_layers
        self.max_iters = max_iters
        self.learning_rate = learning_rate
        self.n_shots = n_shots
        self.verbose = verbose

    def optimize(self, problem, initial_solution=None, **kwargs):
        
        # Set up Ising parameters from the problem.
        h, J, offset, edges = problem.h, problem.J, problem.offset, problem.edges
        n = len(h) # number of qubits

        p = self.num_layers  # number of layers

        # Create the quantum device.
        dev = qml.device("default.qubit", wires=n, shots=self.n_shots)

        # Define the unitary operators for the cost Hamiltonian U_C and mixer U_B.
        def U_C(gamma):
            # For each qubit, apply the local field rotation.
            for i in range(n):
                if (i,) in h and abs(h[(i,)]) > 1e-8:
                    qml.RZ(2 * gamma * h[(i,)], wires=i)
            # For each edge, apply the two-qubit cost unitary.
            for (i, j) in edges:
                if (i, j) in J and abs(J[(i, j)]) > 1e-8:
                    qml.CNOT(wires=[i, j])
                    qml.RZ(2 * gamma * J[(i, j)], wires=j)
                    qml.CNOT(wires=[i, j])

        def U_B(beta):
            # Apply the mixer unitary on all qubits.
            for i in range(n):
                qml.RX(-2 * beta, wires=i)

        @qml.qnode(dev, interface="autograd")
        def circuit(params):
            gammas = params[0]
            betas = params[1]
            # Initial state: uniform superposition via Hadamard gates.
            for i in range(n):
                qml.Hadamard(wires=i)
            # Apply p layers.
            for layer in range(p):
                U_C(gammas[layer])
                U_B(betas[layer])
            # Now, measure the cost by evaluating the expectation of the cost Hamiltonian.
            # We reassemble H_C from our Ising parameters.
            coeffs = []
            obs = []
            for i in range(n):
                if (i,) in h and abs(h[(i,)]) > 1e-8:
                    coeffs.append(h[(i,)])
                    obs.append(qml.PauliZ(i))
            for (i, j) in edges:
                if (i, j) in J and abs(J[(i, j)]) > 1e-8:
                    coeffs.append(J[(i, j)])
                    obs.append(qml.operation.Tensor(qml.PauliZ(i), qml.PauliZ(j)))
            H_C = qml.Hamiltonian(coeffs, obs)
            return qml.expval(H_C)

        # Initialize parameters for gammas and betas.
        # A common initialization for QAOA maxcut is to start with small gamma and beta values.
        init_gammas = np.full((p,), 0.01)
        init_betas = np.full((p,), 0.99)  # e.g., starting near 1 for beta
        params = pnp.array([init_gammas, init_betas], requires_grad=True)

        # Use PennyLane's Adam optimizer.
        opt = qml.AdamOptimizer(stepsize=self.learning_rate)
        for it in range(self.max_iters):
            params, cost = opt.step_and_cost(circuit, params)
            if self.verbose:
                print(f"Iteration {it}: cost = {cost:.6f}")

        best_params = params

        # Define a sampling QNode.
        @qml.qnode(dev, interface="autograd")
        def sample_circuit(params):
            gammas = params[0]
            betas = params[1]
            for i in range(n):
                qml.Hadamard(wires=i)
            for layer in range(p):
                U_C(gammas[layer])
                U_B(betas[layer])
            return qml.sample()

        samples = sample_circuit(best_params)
        # Choose the sample with the lowest classical cost (or use other criteria).
        best_sample = None
        best_energy = float("inf")
        for sample in samples:
            # Convert bitstring (0,1) to spins (+1,-1): here assume 0 -> +1, 1 -> -1.
            spins = [1 if bit == 0 else -1 for bit in sample]
            E = problem.evaluate_solution(spins)
            if E < best_energy:
                best_energy = E
                best_sample = sample

        final_cost = problem.evaluate_solution(best_sample)
        return best_sample.tolist(), final_cost
