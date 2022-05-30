<<<<<<< HEAD
from qiskit_nature.drivers import UnitsType, Molecule
from libs.utils import Quantum_System, SOLVER
from libs.optimize import OPTIMIZER
from qiskit.algorithms.optimizers import COBYLA
import os
import sys
import numpy as np

def hidrogen(r):
    return Molecule(geometry=[["H", [0.0, 0.0, 0.0]], ["H", [0.0, 0.0, r]]])

if __name__ == '__main__':
    args = sys.argv[1:]
    R = np.arange(0.1,2,0.1)
    optimizer = COBYLA()
    Hidrogen = Quantum_System(hidrogen,R[0])
    n_cores = os.cpu_count()

    if args[0].upper() == 'S':
        solver = SOLVER(R,Hidrogen,optimizer,n_cores)
        energies = solver.solve(int(args[1]))

    if args[0].upper() == 'O':
        solver = SOLVER(R,Hidrogen,optimizer,n_cores)
        print(f'\nStart optimazation study for Hidogen molecule')
        print(f'\tNpoints: {len(R)}')
        study = OPTIMIZER(solver,N_max=1,trials=1)
        study.start_study()
        energies = study.BEST_RESULTS
        print(f'\n\nBest number of layers: {study.BEST_PARAMS}\tCost: {study.BEST_RESULT}')
    
    with open('Hidrogen.energies','wb') as f:
        np.save(f,energies)
=======
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
>>>>>>> 1bcbd39a85b0c6bd35fe240ab8c5e8fc5245c3c4
    