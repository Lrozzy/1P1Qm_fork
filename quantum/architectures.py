from typing import Any
import pennylane as qml
from helpers.utils import getIndex
from itertools import combinations

backend = ''
dev = None
all_wires=None
two_comb_wires = None
auto_wires = None
ref_wires = None
ancillary_wires = None
index = None
n_trash_qubits = -1

def initialize(wires:int=4, shots:int=5000, trash_qubits:int=0, backend_name:str='default.qubit'):
    global dev,all_wires, auto_wires, two_comb_wires, ref_wires, ancillary_wires, backend, index, n_trash_qubits
    backend=backend_name
    n_trash_qubits = trash_qubits
    two_comb_wires=list(combinations([i for i in range(wires)],2))
    all_wires=[_ for _ in range(wires+trash_qubits+1)]
    ancillary_wires=all_wires[-1:]
    auto_wires=all_wires[:wires]
    ref_wires=all_wires[wires:wires+trash_qubits] # Do not initialize ref_wires before n_trash_qubits is set
    
    index={'eta':getIndex('particle','eta'),'phi':getIndex('particle','phi'),'pt':getIndex('particle','pt')}

def set_device(shots:int=5000):
    global dev,all_wires,backend
    dev=qml.device(backend,wires=len(all_wires),shots=shots)

def print_params():
    print("\n Sanity check: \n")
    print('all_wires:',all_wires)
    print()
    print('auto_wires:',auto_wires)
    print('two_comb_wires:',two_comb_wires)
    print('ref_wires:',ref_wires)
    print('ancillary_wires:',ancillary_wires)
    print('backend:',backend)
    print('index:',index)
    print('n_trash_qubits:',n_trash_qubits)
    print("\n ############################################## \n")

def circuit(weights,inputs=None):
    # State preparation for all wires
    N = len(auto_wires)  # Assuming wires is a list like [0, 1, ..., N-1]
    # State preparation for all wires
    for w in auto_wires:
        # Variables named according to spherical coordinate system, it's easier to understand :)
        
        zenith = inputs[:,w, index['eta']] # corresponding to eta
        azimuth = inputs[:,w, index['phi']] # corresponding to phi
        radius = inputs[:,w, index['pt']] # corresponding to pt
        # Apply rotation gates modulated by the radius (pt) of the particle, which has been scaled to the range [0,1]
        qml.RY(radius * zenith, wires=w)   
        qml.RZ(radius * azimuth, wires=w)  
        #qml.Rot( 0, radius * zenith, radius * azimuth,wires=w)
    # QAE Circuit

    for phi,theta,omega,i in zip(weights[:N],weights[N:2*N],weights[2*N:],auto_wires):
        qml.Rot(phi,theta,omega,wires=[i]) # perform arbitrary rotation in 3D space instead of RX/RY rotation
    
    for item in two_comb_wires: 
        qml.CNOT(wires=item)
    
    # SWAP test to measure fidelity
    qml.Hadamard(ancillary_wires)
    for ref_wire,trash_wire in zip(ref_wires,auto_wires[-n_trash_qubits:]):
        qml.CSWAP(wires=[ancillary_wires[0], ref_wire, trash_wire])
    qml.Hadamard(ancillary_wires)
    return qml.expval(qml.operation.Tensor(*[qml.PauliZ(i) for i in ancillary_wires]))
   
class QuantumAutoencoder:
    def __init__(self, wires:int=4, shots:int=5000, trash_qubits:int=0, backend_name:str='default.qubit'):
        initialize(wires=wires, shots=shots, trash_qubits=trash_qubits, backend_name=backend_name)
        set_device(shots=shots)
    def fetch_circuit(self):
        global dev
        return qml.QNode(circuit,dev)

def fetch_device():
        global dev
        return dev