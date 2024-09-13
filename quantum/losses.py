import pennylane.numpy as nnp

def semi_classical_cost(weights,inputs=None,quantum_cost=None,return_fid=False):
    if return_fid:
        fid=quantum_cost(weights,inputs)
        cost=1.-fid
        return 100.*nnp.array(cost),100.*nnp.array(fid)
    
    # the minus sign is to maximize the fidelity between the trash and reference states - which implies that the output and input states are close to each other
    cost=nnp.array(1-quantum_cost(weights,inputs),requires_grad=False)#np.array([-quantum_cost(weights,item) for item in inputs]).mean()
    return 100.*nnp.array(cost)