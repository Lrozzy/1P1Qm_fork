from argparse import ArgumentParser
import glob
from itertools import combinations
import helpers.utils as ut
import pennylane as qml
import matplotlib.pyplot as plt
import case_reader as cr
import pickle
import os,pathlib
import matplotlib.pyplot as plt
import matplotlib;matplotlib.use('Agg')
from tqdm import tqdm
import numpy as nnp
parser=ArgumentParser(description='select options to train quantum autoencoder')
parser.add_argument('--seed',default=9999,type=int,help='Some number to index the run')
args=parser.parse_args()

import pennylane.numpy as np


def Unpickle(path=None):
    with open(path,'rb') as f:
        return_object=pickle.load(f)
    return return_object

def semi_classical_cost(weights,inputs=None,quantum_cost=None,return_fid=False):
    import pdb;pdb.set_trace()
    if return_fid:
        fid=quantum_cost(weights,inputs)
        cost=1.-fid
        return nnp.array(cost),nnp.array(fid)
    
    # the minus sign is to maximize the fidelity between the trash and reference states - which implies that the output and input states are close to each other
    cost=np.array(1-quantum_cost(weights,inputs),requires_grad=False)#np.array([-quantum_cost(weights,item) for item in inputs]).mean()
    return nnp.array(cost)



save_dir=os.path.join(ut.path_dict['QAE_save'],str(args.seed))
plot_dir=os.path.join(save_dir,'plots')
pathlib.Path(plot_dir).mkdir(parents=True,exist_ok=True)
assert os.path.isfile(os.path.join(save_dir,'args.pickle')),'args.pickle not found in: '+save_dir

test_args=Unpickle(os.path.join(save_dir,'args.pickle'))

dev=qml.device('lightning.kokkos',wires=test_args.wires+test_args.trash_qubits+1,shots=test_args.shots)
two_comb_wires=list(combinations([i for i in range(test_args.wires)],2))
all_wires=[_ for _ in range(test_args.wires+test_args.trash_qubits+1)]
ancillary_wires=all_wires[-1:]
auto_wires=all_wires[:test_args.wires]
ref_wires=all_wires[test_args.wires:test_args.wires+test_args.trash_qubits]
    
index={'eta':ut.getIndex('particle','eta'),'phi':ut.getIndex('particle','phi'),'pt':ut.getIndex('particle','pt')}
dev=qml.device('lightning.kokkos',wires=len(all_wires),shots=test_args.shots)


@qml.qnode(dev,interface=test_args.backend)
def circuit(weights,inputs=None):

    # State preparation for all wires
    N = len(auto_wires)  # Assuming wires is a list like [0, 1, ..., N-1]
    # State preparation for all wires
    for w in auto_wires:
        # Variables named according to spherical coordinate system, it's easier to understand :)
        #import pdb;pdb.set_trace()
        zenith = inputs[w, index['eta']] # corresponding to eta
        azimuth = inputs[w, index['phi']] # corresponding to phi
        radius = inputs[w, index['pt']] # corresponding to pt
        #import pdb;pdb.set_trace()
        #print('HAA')
        # Apply rotation gates modulated by the radius (pt) of the particle, which has been scaled to the range [0,1]
        qml.RY(radius * zenith, wires=w)   
        qml.RZ(radius * azimuth, wires=w)  
        #qml.Rot( 0, radius * zenith, radius * azimuth,wires=w)
    # QAE Circuit
    
    for item,i in zip(weights,auto_wires):
        qml.RX(item/2.,wires=[i]) # change from RY to RX rotation for the actual trainable weights
        
    for item in two_comb_wires: 
        qml.CNOT(wires=item)
    
    # SWAP test to measure fidelity
    qml.Hadamard(ancillary_wires)
    for ref_wire,trash_wire in zip(ref_wires,auto_wires[-test_args.trash_qubits:]):
        qml.CSWAP(wires=[ancillary_wires[0], ref_wire, trash_wire])
    qml.Hadamard(ancillary_wires)
    return qml.expval(qml.operation.Tensor(*[qml.PauliZ(i) for i in ancillary_wires]))
#@qml.qnode(dev,interface=args.backend)

model_path=os.path.join(save_dir,'trained_model.pickle')


assert os.path.isfile(model_path),'Model not found at: '+model_path

print ('Testing with model at:',model_path)
history=Unpickle(path=os.path.join(save_dir,'history.pickle'))

#print (history)
val_loss=history['val']
epoch=np.argmin(val_loss)
print (epoch,val_loss[epoch])
dictionary=Unpickle(model_path)
weights=np.array(dictionary['weights'],requires_grad=False)

test_loader=cr.CASEDelphesDataLoader(filelist=sorted(glob.glob(ut.path_dict['QCD_lib']+'/*.h5')),batch_size=test_args.batch_size,input_shape=(len(auto_wires),3),train=False,max_samples=1e5)  # Shuffle is set to False
sig_loader=cr.CASEDelphesDataLoader(filelist=sorted(glob.glob(ut.path_dict['grav_2p5_narrow']+'/*.h5')),batch_size=test_args.batch_size,input_shape=(len(auto_wires),3),train=False,max_samples=1e5)  # Shuffle is set to False

test_fid=[]
sig_fid=[]

for data,label in tqdm(test_loader):
    import pdb;pdb.set_trace()
    cost,fid=semi_classical_cost(weights,inputs=data,quantum_cost=circuit,return_fid=True)
    test_fid.append(fid)
for data,label in tqdm(sig_loader):
    cost,fid=semi_classical_cost(weights,inputs=data,quantum_cost=circuit,return_fid=True)
    sig_fid.append(fid)    
sig_fid=nnp.concatenate(sig_fid,axis=0)
test_fid=nnp.concatenate(test_fid,axis=0)

# test_fid=np.array(test_fid)
# sig_fid=np.array(sig_fid)
import pdb;pdb.set_trace()
plt.hist(test_fid,bins=150,alpha=0.5,label='QCD',range=[0,1.5])
plt.hist(sig_fid,bins=150,alpha=0.5,label='$M_{grav}=2.5$ TeV',range=[0,1.5])

plt.xlabel('Quantum Fidelity: $<T|R>$')
plt.ylabel('No. of events')
plt.legend(loc='upper right')
plt.savefig(os.path.join(plot_dir,'fidelity_hist.png'))
