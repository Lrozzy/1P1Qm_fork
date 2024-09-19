import pennylane.numpy as np

def semi_classical_cost(weights,inputs=None,quantum_cost=None,return_fid=False):
    if return_fid:
        fid=quantum_cost(weights,inputs)
        #fid=np.sqrt(fid)
        cost=1.-fid
        return 100.*np.array(cost),100.*np.array(fid)
    
    # the minus sign is to maximize the fidelity between the trash and reference states - which implies that the output and input states are close to each other
    fid=quantum_cost(weights,inputs)
    #fid=np.sqrt(fid)
    cost=1-fid#np.array([-quantum_cost(weights,item) for item in inputs]).mean()
    return 100.*np.array(cost,requires_grad=False)

def batch_semi_classical_cost(weights,inputs=None,quantum_cost=None,return_fid=False):
    if return_fid:
        fid=quantum_cost(weights,inputs)
        #fid=np.sqrt(fid)
        cost=1.-fid
        return np.array(100.*cost,requires_grad=True).mean(),np.array(100.*fid,requires_grad=True).mean()
    # the minus sign is to maximize the fidelity between the trash and reference states - which implies that the output and input states are close to each other
    fid=quantum_cost(weights,inputs)
    #fid=np.sqrt(fid)
    cost=1-fid
    batched_average_cost=100.*(np.array(cost,requires_grad=True).mean())#np.array([-quantum_cost(weights,item) for item in inputs]).mean()
    return batched_average_cost