import pennylane.numpy as np
import sys
def mean_squared_error(predictions,targets):
    return np.mean((predictions-targets)**2)

def binary_cross_entropy(labels, probs,bias=None):
    # Ensure that probabilities are bounded between (0, 1)
    probs = np.clip(probs, 1e-15, 1 - 1e-15)  # Avoid log(0) which would cause issues
    # Compute Binary Cross-Entropy
    loss = -np.mean(labels * np.log(probs) + (1 - labels) * np.log(1 - probs))
    # if bias is not None:
    #     loss+=bias
    return loss

def sigmoid(x):
    return 1/(1+np.exp(-x))

def semi_classical_cost(weights,inputs=None,quantum_circuit=None,return_fid=False):
    if return_fid:
        fid=quantum_circuit(weights,inputs)
        #fid=np.sqrt(fid)
        cost=1.-fid
        return 100.*np.array(cost),100.*np.array(fid)
    
    # the minus sign is to maximize the fidelity between the trash and reference states - which implies that the output and input states are close to each other
    fid=quantum_circuit(weights,inputs)
    #fid=np.sqrt(fid)
    cost=1-fid#np.array([-quantum_circuit(weights,item) for item in inputs]).mean()
    return 100.*np.array(cost,requires_grad=False)

def batch_semi_classical_cost(weights,inputs=None,quantum_circuit=None,return_fid=False):
    if return_fid:
        fid=quantum_circuit(weights,inputs)
        #fid=np.sqrt(fid)
        cost=1.-fid
        return np.array(100.*cost,requires_grad=True).mean(),np.array(100.*fid,requires_grad=True).mean()
    # the minus sign is to maximize the fidelity between the trash and reference states - which implies that the output and input states are close to each other
    fid=quantum_circuit(weights,inputs)
    #fid=np.sqrt(fid)
    cost=1-fid
    batched_average_cost=100.*(np.array(cost,requires_grad=True).mean())#np.array([-quantum_circuit(weights,item) for item in inputs]).mean()
    return batched_average_cost

def VQC_cost(weights,inputs=None,quantum_circuit=None,labels=None,return_scores=False,val=False,loss_type='BCE'):
    bias=weights[-1]
    exp_vals=np.array(quantum_circuit(weights,inputs),requires_grad=True) # n_qubits x batch_size
    #exp_vals=0.5*(1+np.mean(exp_vals,axis=0,rquires_grad=True)) # reduce over the first axis, which is the number of wires
    #score = expvals
    score=(exp_vals+bias)
    #score=1+0.5*exp_vals#+bias
    
    if loss_type=='BCE':
        loss_fn=binary_cross_entropy(labels,score)
    elif loss_type=='MSE':
        loss_fn=mean_squared_error(labels,score)
    else:
        sys.exit(-1)
    if return_scores:
        if val:
            score=np.mean(exp_vals)
        return loss_fn, score
    return np.array(loss_fn,requires_grad=True)

def probabilistic_loss(weights,inputs=None,quantum_circuit=None,labels=None,return_scores=False,val=False,loss_type='BCE'):
    batch_size=len(labels)
    probs=np.array(quantum_circuit(weights,inputs),requires_grad=True) # n_qubits x batch_size
    
    if batch_size==1:
        scores=1-probs[labels[0]]
    else:
        scores=1-probs[np.arange(probs.shape[0]), labels]
    loss_fn=np.mean(scores)
        
    if return_scores:
        if val:
            score=np.mean(scores)
        return loss_fn, score
    return np.array(loss_fn,requires_grad=True)
    