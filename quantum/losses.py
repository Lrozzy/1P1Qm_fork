import pennylane.numpy as nnp

def semi_classical_cost(weights,inputs=None,quantum_cost=None,return_fid=False):
    if return_fid:
        fid=quantum_cost(weights,inputs)
        cost=1.-fid
        return 100.*nnp.array(cost),100.*nnp.array(fid)
    
    # the minus sign is to maximize the fidelity between the trash and reference states - which implies that the output and input states are close to each other
    cost=nnp.array(1-quantum_cost(weights,inputs),requires_grad=False)#np.array([-quantum_cost(weights,item) for item in inputs]).mean()
    return 100.*nnp.array(cost)

def batch_semi_classical_cost(weights,inputs=None,quantum_cost=None,return_fid=False):
    if return_fid:
        cost,fid=[],[]
        #for item in inputs:
        #    fid.append(quantum_cost(weights,item))
        #    cost.append(1-fid[-1])
        fid=quantum_cost(weights,inputs)
        cost=1.-fid
        return nnp.array(100.*cost,requires_grad=True).mean(),nnp.array(100.*fid,requires_grad=True).mean()
    # the minus sign is to maximize the fidelity between the trash and reference states - which implies that the output and input states are close to each other
    batched_average_cost=100.*(nnp.array(1-quantum_cost(weights,inputs),requires_grad=True).mean())#np.array([-quantum_cost(weights,item) for item in inputs]).mean()
    return batched_average_cost