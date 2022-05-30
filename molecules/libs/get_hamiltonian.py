from qiskit_nature.drivers import UnitsType, Molecule
from qiskit_nature.drivers.second_quantization import (
    ElectronicStructureDriverType,
    ElectronicStructureMoleculeDriver,
)
from qiskit_nature.problems.second_quantization import ElectronicStructureProblem
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.mappers.second_quantization import JordanWignerMapper, ParityMapper
from qiskit.algorithms import NumPyMinimumEigensolver
from qiskit_nature.algorithms import GroundStateEigensolver
import numpy as np
import pickle
import matplotlib.pyplot as plt

def get_H(ops):
    coeffs = np.real(ops.coeffs)
    paulis = [p.to_label() for p in ops.primitive.paulis]
    return list(zip(coeffs,paulis))


def get_hamiltonian(geometry):
    molecule = Molecule(geometry=geometry)
    driver = ElectronicStructureMoleculeDriver(molecule, basis="sto3g", driver_type=ElectronicStructureDriverType.PYSCF)
    es_problem = ElectronicStructureProblem(driver)
    qubit_converter = QubitConverter(mapper=ParityMapper(), two_qubit_reduction=True)
    second_q_op = es_problem.second_q_ops()
    qubit_op = qubit_converter.convert(second_q_op[0], num_particles=es_problem.num_particles)
    Hamiltonian = get_H(qubit_op)
    return Hamiltonian

def get_hamiltonians():
    dist = np.linspace(0.1,2,100)
    hamiltonians = []
    for i in range(len(dist)):
        geometry = geometry=[["H", [0.0, 0.0, 0.0]], ["H", [0.0, 0.0, dist[i]]]]
        hamiltonian = get_hamiltonian(geometry)
        hamiltonians.append(hamiltonian)
    return hamiltonians

def save_hamiltonians(hamiltonians):
    file_name = "hamiltonians.pkl"
    open_file = open(file_name, "wb")
    pickle.dump(hamiltonians, open_file)
    open_file.close()

def load_hamiltonians(file):
    open_file = open(file, "rb")
    loaded_list = pickle.load(open_file)
    open_file.close()
    return loaded_list

#def filter_hamiltonians(hamiltonians):
#    for i in range(len(hamiltonians)):


