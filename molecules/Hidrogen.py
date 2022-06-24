from qiskit_nature.drivers import UnitsType, Molecule
from libs.utils import Quantum_System, SOLVER
from libs.optimize import OPTIMIZER
from libs.headers import header, footer
from qiskit.algorithms.optimizers import COBYLA
import os
import sys
import time
import numpy as np

def hidrogen(r):
    return Molecule(geometry=[["H", [0.0, 0.0, 0.0]], ["H", [0.0, 0.0, r]]], multiplicity=1, charge=0,)

if __name__ == '__main__':

    args = sys.argv[1:]
    R = np.arange(1.9,2,0.1)
    optimizer = COBYLA()
    n_cores = os.cpu_count()
    t_start = time.time()

    parameters = {'Start Time': time.ctime(), 
                  'N Cores': n_cores,
                  'N points': len(R)}

    if args[0].upper() == 'S':
        if len(args[1:]) < 2:
            print('Not enough arguments')
            exit()
        parameters['Operation'] = 'Solve'
        N_layers = int(args[1])
        mixer = args[2]
        parameters['N Layers'] = N_layers
        type_solution = f'Solve_{mixer}{N_layers}'
        header('Hidrogen', parameters)
        solver = SOLVER(R,hidrogen,optimizer,n_cores,mixer)
        energies = solver.solve(N_layers)
    
    if args[0].upper() == 'V':
        parameters['Operation'] = 'VQE'
        header('Hidrogen', parameters)
        type_solution = f'VQE'
        energies = []
        for r in R:
            H = Quantum_System(hidrogen,r)
            energies.append([np.real(H.calc_VQE()),H.res.eigenenergies[0]])
            print(energies[-1])
        energies = np.array(energies)

    if args[0].upper() == 'O':
        parameters['Operation'] = 'Optimize'
        header('Hidrogen', parameters)
        solver = SOLVER(R,hidrogen,optimizer,n_cores)
        print(f'\nStart optimazation study for Hidogen molecule')
        print(f'\tNpoints: {len(R)}')
        study = OPTIMIZER(solver,N_max=15,trials=100)
        study.start_study()
        energies = study.BEST_RESULTS
        print(f'\n\nBest number of layers: {study.BEST_PARAMS}\tCost: {study.BEST_RESULT}')
    
    footer(t_start,time.time())

    with open(f'Hidrogen_{type_solution}.energies','wb') as f:
        np.save(f,energies)
    