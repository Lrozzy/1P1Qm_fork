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

def batch_quantum_cost(weights,inputs=None,quantum_cost=None,return_fid=False):
    probs=np.array(quantum_cost(weights,inputs))[...,0] # the first element is the probability of the |0> state, which is a function of the fidelity
    # Prob(0) = 0.5*(1+fid**2)  --> We need to invert this
    fid = np.sqrt(2*probs-1)
    cost=1-fid
    if return_fid:
        return np.mean(100.*cost),np.mean(100.*fid)
    # the minus sign is to maximize the fidelity between the trash and reference states - which implies that the output and input states are close to each other
    #fid=np.sqrt(fid)
    
    batched_average_cost=100.*np.mean(cost)#np.array([-quantum_cost(weights,item) for item in inputs]).mean()
    return batched_average_cost


def quantum_cost(weights,inputs=None,quantum_cost=None,return_fid=False):
    probs=np.array(quantum_cost(weights,inputs))[...,0] # the first element is the probability of the |0> state, which is a function of the fidelity
    # Prob(0) = 0.5*(1+fid**2)  --> We need to invert this
    fid = np.sqrt(2*probs-1)
    cost=1-fid
    if return_fid:
        return np.mean(100.*cost,axis=0,requires_grad=False),np.mean(100.*fid,axis=0,requires_grad=False)
    # the minus sign is to maximize the fidelity between the trash and reference states - which implies that the output and input states are close to each other
    #fid=np.sqrt(fid)
    
    batched_average_cost=100.*np.mean(cost,axis=0,requires_grad=False)#np.array([-quantum_cost(weights,item) for item in inputs]).mean()
    return batched_average_cost