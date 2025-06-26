import strawberryfields as sf
from strawberryfields.ops import Sgate, Dgate, BSgate, CXgate
import numpy as np

def symbolic_circuit(prog, wires, weights):
    """
    Constructs a symbolic quantum circuit for the 1P1Qm model.
    
    Args:
        prog: The Strawberry Fields program object.
        wires: Number of wires in the circuit.
        weights: A dictionary containing circuit parameters.  
    """
    # Extract non-trainable parameters from weights
    eta = [weights[f'eta_{i}'] for i in range(wires)]
    pt = [weights[f'pt_{i}'] for i in range(wires)]
    phi = [weights[f'phi_{i}'] for i in range(wires)]

    # Extract trainable parameters
    s_scale = weights['s_scale']    
    squeeze_mag = [weights[f'squeeze_mag_{i}'] for i in range(wires)]
    squeeze_phase = [weights[f'squeeze_phase_{i}'] for i in range(wires)]
    disp_mag = [weights[f'disp_mag_{i}'] for i in range(wires)]
    disp_phase = [weights[f'disp_phase_{i}'] for i in range(wires)]
    
    with prog.context as q:
        scale = 10.0 / (1.0 + sf.math.exp(-s_scale)) + 0.01
        for w in range(wires):
            Sgate(eta[w], pt[w]*phi[w]/2) | q[w] # Encode eta and phi
            Dgate(scale*pt[w], eta[w])    | q[w] # Encode scale, pt, and eta
            
        for a,b in [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]:
            CXgate(1.0) | (q[a], q[b]) # Entangle wires
        all_wires_list = list(range(wires))
        for i in range(wires):
            idx1 = all_wires_list[i]
            idx2 = all_wires_list[(i + 1) % wires]
            BSgate(np.pi / 4.0, np.pi / 2.0) | (q[idx1], q[idx2]) # Further entanglement
            
        for w in range(wires):
            Sgate(squeeze_mag[w], squeeze_phase[w]) | q[w] # Encode trainable parameters
            Dgate(disp_mag[w], disp_phase[w]) | q[w] # Encode trainable parameters
    
    return prog