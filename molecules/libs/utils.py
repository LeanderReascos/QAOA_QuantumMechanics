from qiskit_nature.drivers.second_quantization import (
    ElectronicStructureDriverType,
    ElectronicStructureMoleculeDriver,
)
from qiskit_nature.problems.second_quantization import ElectronicStructureProblem
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.mappers.second_quantization import JordanWignerMapper, ParityMapper
from qiskit.algorithms import NumPyMinimumEigensolver
from qiskit_nature.algorithms import GroundStateEigensolver

from qiskit import QuantumCircuit,QuantumRegister, Aer, transpile
import numpy as np

from multiprocessing import Process, shared_memory

def execute_circuit(qc, shots=1024, device=None):
    device = Aer.get_backend('qasm_simulator') if device is None else device
    transpiled_circuit = transpile(qc, device)
    counts = device.run(transpiled_circuit, shots=shots).result().get_counts()
    return counts

def basis_states_probs(counts):
    n = len(list(counts.keys())[0])
    N = sum(list(counts.values()))
    return np.array([counts[np.binary_repr(vals,n)]/N if counts.get(np.binary_repr(vals,n)) is not None else 0 for vals in range(2**n)])

def get_H(ops):
    coeffs = np.real(ops.coeffs)
    paulis = [p.to_label() for p in ops.primitive.paulis]
    return list(zip(coeffs,paulis))

def get_basis_change(p):
    qc = QuantumCircuit(1)
    if p == 'X':
            qc.h(0)
    if p == 'Y':
        qc.u(np.pi/2,np.pi/2,np.pi/2,0)
    return qc

def get_rGate(n_qubits,h):
    q_register = QuantumRegister(n_qubits)
    qc = QuantumCircuit(q_register)
    contrl_index = [i for i,c in enumerate(h[::-1]) if c != 'I']
    if len(contrl_index) == 0:
        return qc,-1
    for i in contrl_index:
        qc = qc.compose(get_basis_change(h[::-1][i]),[i])
    for i,k in enumerate(contrl_index[::-1][:-1]):
        qc.cx(k,contrl_index[::-1][i+1])
    return qc,contrl_index[0]

def compute_eh(n_qubits,c,h,alpha):
    q_register = QuantumRegister(n_qubits)
    qc = QuantumCircuit(q_register)
    rGate,qr = get_rGate(n_qubits,h)
    if qr == -1:
        return qc
    qc = qc.compose(rGate,range(n_qubits))
    qc.rz(c*alpha,qr)
    qc = qc.compose(rGate.inverse(),range(n_qubits))
    return qc


class Quantum_System:
    def __init__(self,molecule,*args):
        self.molecule = molecule(*args)
        driver = ElectronicStructureMoleculeDriver(self.molecule, basis="sto3g", driver_type=ElectronicStructureDriverType.PYSCF)
        es_problem = ElectronicStructureProblem(driver)
        qubit_converter = QubitConverter(mapper=ParityMapper(), two_qubit_reduction=True)
        self.second_q_op = es_problem.second_q_ops()
        self.qubit_op = qubit_converter.convert(self.second_q_op[0], num_particles=es_problem.num_particles)
        self.Hamiltonian = get_H(self.qubit_op)
        self.num_qubits = len(self.Hamiltonian[0][1])
        numpy_solver = NumPyMinimumEigensolver()
        calc = GroundStateEigensolver(qubit_converter, numpy_solver)
        self.res = calc.solve(es_problem) 
    
    def get_EvolutionOperator(self):
        def get_eH(alpha):
            num_qubits = len(self.Hamiltonian[0][1])
            qregister = QuantumRegister(num_qubits)
            qc = QuantumCircuit(qregister)
            for c,h in self.Hamiltonian:
                qc = qc.compose(compute_eh(num_qubits,c,h,alpha),range(num_qubits))
            return qc
        return get_eH
    
    def expval_HamiltonianOperator(self,state_preparation,shots=2048):
        expval = 0
        for c,h in self.Hamiltonian:
            quantum_register = QuantumRegister(self.num_qubits)
            qc = QuantumCircuit(quantum_register)
            qc = qc.compose(state_preparation,quantum_register)
            eig_vals = np.ones(4)
            for i,p in enumerate(h[::-1]):
                if p != 'Z' and p != 'I':
                    qc = qc.compose(get_basis_change(p),[i])
                if p != 'I':
                    eig_vals *= np.array([(-1)**int(np.binary_repr(n,2)[::-1][i]) for n in range(4)])
            qc.measure_all()
            expval += c*np.sum(basis_states_probs(execute_circuit(qc,shots=shots))*eig_vals)
        
        return expval
    
class QAOA:
    def __init__(self,num_qubits,cost_function,e_H,mixer='Uniform',N_layers=1):
        self.num_qubits = num_qubits
        self.mixer = mixer if type(mixer) is not str else self.get_mixer(mixer)
        self.N = N_layers
        self.cost_function = cost_function
        self.eH = e_H
        self.BEST_PARAMS = None
        self.BEST_RESULT = np.inf
        self.HISTORY = []
    
    def get_mixer(self,mixer):
        qc_init = QuantumCircuit(self.num_qubits)
        [qc_init.h(i) for i in range(self.num_qubits)]
        def qc_mix(beta):
            qc_mixer = QuantumCircuit(self.num_qubits)
            [qc_mixer.rx(beta,i) for i in range(self.num_qubits)]
            return qc_mixer
        return qc_init,qc_mix

    def QAOA(self,parameters):
        q_register = QuantumRegister(self.num_qubits,'q')
        qc = QuantumCircuit(q_register,name='QAOA')
        qc = qc.compose(self.mixer[0],q_register)
        for p in range(self.N):
            alpha,beta = parameters[2*p],parameters[p+1]
            qc = qc.compose(self.eH(alpha),q_register)
            qc = qc.compose(self.mixer[1](beta),q_register)
        return qc
    
    def objective_function(self,parameters):
        result = self.cost_function(self.QAOA(parameters))
        self.HISTORY.append(result)
        if result < self.BEST_RESULT:
            self.BEST_RESULT = result
            self.BEST_PARAMS = parameters
        return result


class SOLVER:
    def __init__(self,R,molecule_function,optimizer,n_cores):
        self.R = R
        self.molecule = molecule_function
        self.optimizer = optimizer
        self.n_cores = n_cores
    
    def solve(self,N=1):
        self.N = N
        print(f' N Layers: {N}')
        def simulate(smm,i_start,i_end):
            R = np.ndarray((len(self.R),2), dtype=np.float64, buffer=smm.buf)
            params = np.random.random(2*self.N)*2*np.pi
            for i,r in enumerate(self.R[i_start:i_end]):
                molecule = Quantum_System(self.molecule,r)
                qaoa = QAOA(molecule.num_qubits,molecule.expval_HamiltonianOperator,molecule.get_EvolutionOperator(),N_layers=self.N)
                opt_var, opt_value, _ = self.optimizer.optimize(2*qaoa.N, qaoa.objective_function, initial_point=params)
                R[i_start+i][0]=qaoa.BEST_RESULT
                R[i_start+i][1]=molecule.res.eigenenergies[0]
                params = qaoa.BEST_PARAMS
                print(f'    i: {i_start+i} - r: {r:.2f} Energy: {qaoa.BEST_RESULT,molecule.res.eigenenergies[0]}')
            del R
        self.results = self.create_process(simulate)
        return self.results

    def create_process(self,f):
        results_ =  np.zeros((len(self.R),2),dtype=np.float64)
        smm = shared_memory.SharedMemory(create=True,size=results_.nbytes)
        results = np.ndarray(results_.shape, dtype=results_.dtype, buffer=smm.buf)
        del results_
        process = []
        self.n_cores = self.n_cores if self.n_cores < len(self.R) else len(self.R)
        print(f'\t\nCores: {self.n_cores}')
        rs = len(self.R)//self.n_cores
        for n in range(self.n_cores):
            ns = (n+1)*rs if n < self.n_cores-1 else len(self.R)
            p = Process(target=f, args=(smm,n*rs,ns))
            p.start()
            process.append(p)
        
        while len(process)>0:
            p = process.pop(0)
            p.join()
        res = results.copy()
        del results
        smm.close()
        smm.unlink()
        return res