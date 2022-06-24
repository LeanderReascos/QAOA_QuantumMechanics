from cProfile import label
import numpy as np
import matplotlib.pyplot as plt

def open_energies(filename):
    with open('Energies/'+filename, 'rb') as f:
        energies = np.load(f)
    return energies

def plot_energies(mixer,N):
    energies = open_energies(f'Hidrogen_Solve_{mixer}{N}.energies')
    fig, ax = plt.subplots()
    R = np.arange(0.1,2,0.1)
    ax.plot(R,energies[:,0], label='$QAOA$')
    ax.plot(R,energies[:,1], label='$Numpy$')
    ax.legend()
    ax.set_xlabel('$Internuclear$ $Distance$ $A$')
    ax.set_ylabel('$Energy$ $Hartreefock$')
    ax.set_title(f'${mixer}$ ${N}$ $layers$')
    return fig,ax

def evaluate_energies(energies):
    return np.mean(np.abs(energies[:,0]-energies[:,1]))

def compare_layers():
    fig,ax = plt.subplots()
    vqe = evaluate_energies(open_energies('Hidrogen_VQE.energies'))
    N = np.arange(1,21)
    ax.plot(N,[vqe]*20, label ='$VQE$')
    for mixer in ['Rx','Ry']:
        errors = np.empty(len(N),float)
        for n in N:
            E = open_energies(f'Hidrogen_Solve_{mixer}{n}.energies')
            errors[n-1] = evaluate_energies(E)
        ax.set_xlabel(r'$N$ $Layers$')
        ax.set_ylabel(r'$\overline{Error}$')
        ax.plot(N,errors,label=f'$QAOA$-${mixer}$')
    ax.set_yscale('log')
    ax.legend()
    return fig,ax

def error_layers():
    fig,ax = plt.subplots()
    vqe = open_energies('Hidrogen_VQE.energies')
    vqe_error = np.abs(vqe[:,0]-vqe[:,1])
    R = np.arange(0.1,2,0.1)
    ax.plot(R,vqe_error,label='$VQE$')
    for mixer in ['Rx','Ry']:
        errors = np.zeros(19)
        for n in range(1,21):
            E = open_energies(f'Hidrogen_Solve_{mixer}{n}.energies')
            errors += np.abs(E[:,0]-E[:,1])
        ax.plot(R,errors/20,label=f'$QAOA$-${mixer}$')
        ax.set_xlabel(r'$N$ $Layers$')
        ax.set_ylabel(r'$\overline{Error}$')
    ax.set_yscale('log')
    ax.legend()
    return fig,ax