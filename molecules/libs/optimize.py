from libs.utils import SOLVER
import numpy as np
import optuna
from multiprocessing import Process, shared_memory

class OPTIMIZER:
    def __init__(self,solver:SOLVER,trials=100,timeout=3600,N_min=1,N_max=1):
        self.solver = solver
        self.TRIALS = trials
        self.TIMEOUT = timeout
        self.N_min = N_min
        self.N_max = N_max
        
        with open('Hidrogen_VQE.energies','rb') as f:
            self.energies = np.load(f)
        self.vqe_result = np.mean(np.abs(self.energies[:,0]-self.energies[:,1]))
        
        self.BEST_RESULT = np.inf
        self.BEST_PARAMS = None
        self.BEST_RESULTS = None

    def objective(self,trial):
        N = trial.suggest_int("N_layers", self.N_min, self.N_max)
        results = self.solver.solve(N)
        result = np.mean(np.abs(results[:,0]-results[:,1]))
        result = np.abs(result-self.vqe_result)

        if result < self.BEST_RESULT:
            print(f'\t -- VQE: {self.vqe_result} QAOA: {result}')
            self.BEST_RESULT = result
            self.BEST_RESULTS = results
            self.BEST_PARAMS = N
        return result

    def start_study(self):
        study = optuna.create_study(
            sampler=optuna.samplers.TPESampler(multivariate=True),
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
        )
        study.optimize(self.objective, self.TRIALS, self.TIMEOUT, show_progress_bar=True)
