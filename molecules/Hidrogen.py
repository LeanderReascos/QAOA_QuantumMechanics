from libs.utils import Quantum_System, QAOA
from qiskit.algorithms.optimizers import COBYLA, ADAM,SPSA
import numpy as np
#import optuna


if __name__ == '__main__':
    N = 5
    R = np.arange(0.1,2,0.1)
    params = np.random.random(2*N)*2*np.pi
    energies = []
    for r in R:
        hidrogen = Quantum_System([["H", [0.0, 0.0, 0.0]], ["H", [0.0, 0.0, r]]])
        qaoa = QAOA(hidrogen.num_qubits,hidrogen.expval_HamiltonianOperator,hidrogen.get_EvolutionOperator(),N_layers=N)
        #optimizer = COBYLA()
        optimizer = SPSA()
        opt_var, opt_value, _ = optimizer.optimize(2*qaoa.N, qaoa.objective_function, initial_point=params)
        energies.append([qaoa.BEST_RESULT,hidrogen.res.eigenenergies[0]])
        params = qaoa.BEST_PARAMS
        print(qaoa.BEST_RESULT,hidrogen.res.eigenenergies)
    
    energies = np.array(energies)
    with open('Hidrogen_rx_5layers_spsa.energies','wb') as f:
        np.save(f,energies)
    