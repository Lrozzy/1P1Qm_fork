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
from sklearn.metrics import roc_curve,roc_auc_score
from sklearn.preprocessing import MinMaxScaler

parser=ArgumentParser(description='select options to train quantum autoencoder')
parser.add_argument('--seed',default=9999,type=int,help='Some number to index the run')
parser.add_argument('--read_n',default=2e4,type=int,help='No. of test events to read in')
parser.add_argument('--epoch_n',default=1,type=int,help='If you want to load some checkpoint at epoch N')
parser.add_argument('--dump',default=False,action='store_true')
parser.add_argument('--load',default=False,action='store_true')
args=parser.parse_args()

import pennylane.numpy as np

def Unpickle(path=None):
    with open(path,'rb') as f:
        return_object=pickle.load(f)
    return return_object

def semi_classical_cost(weights,inputs=None,quantum_cost=None,return_fid=False):
    if return_fid:
        fid=quantum_cost(weights,inputs)
        cost=1.-fid
        return 100.*nnp.array(cost),100.*nnp.array(fid)
    
    # the minus sign is to maximize the fidelity between the trash and reference states - which implies that the output and input states are close to each other
    cost=np.array(1-quantum_cost(weights,inputs),requires_grad=False)#np.array([-quantum_cost(weights,item) for item in inputs]).mean()
    return 100.*nnp.array(cost)

read_n=args.read_n
ceph_dir='/ceph/abal/QML/'
dump_dir=os.path.join(ceph_dir,'dumps',str(args.seed))
pathlib.Path(dump_dir).mkdir(parents=True,exist_ok=True)
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
dev=qml.device('lightning.qubit',wires=len(all_wires),shots=test_args.shots)


@qml.qnode(dev,interface=test_args.backend)
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
    for ref_wire,trash_wire in zip(ref_wires,auto_wires[-test_args.trash_qubits:]):
        qml.CSWAP(wires=[ancillary_wires[0], ref_wire, trash_wire])
    qml.Hadamard(ancillary_wires)
    return qml.expval(qml.operation.Tensor(*[qml.PauliZ(i) for i in ancillary_wires]))

try:
    model_path=os.path.join(save_dir,'trained_model.pickle')
    assert os.path.isfile(model_path),'Model not found at: '+model_path
except:
    model_path=os.path.join(save_dir,'checkpoints',f'ep03.pickle')


print ('Testing with model at:',model_path)
try:
    history=Unpickle(path=os.path.join(save_dir,'history.pickle'))

    #print (history)
    val_loss=history['val']
    epoch=nnp.argmin(val_loss)
    print (epoch,val_loss[epoch])
except:
    print("Did not find history file. Skipping.")
dictionary=Unpickle(model_path)
weights=np.array(dictionary['weights'],requires_grad=False)

test_loader=cr.CASEDelphesDataLoader(filelist=sorted(glob.glob(ut.path_dict['QCD_lib']+'/*.h5')),batch_size=test_args.batch_size,input_shape=(len(auto_wires),3),train=False,max_samples=read_n)  # Shuffle is set to False
sig_loader=cr.CASEDelphesDataLoader(filelist=sorted(glob.glob(ut.path_dict['grav_4p5_narrow']+'/*.h5')),batch_size=test_args.batch_size,input_shape=(len(auto_wires),3),train=False,max_samples=read_n)  # Shuffle is set to False

test_fid=[]
sig_fid=[]
test_cost=[]
sig_cost=[]
if args.load:
    test_fid=nnp.load(os.path.join(dump_dir,'test_fidelities.npy'))
    sig_fid=nnp.load(os.path.join(dump_dir,'sig_fidelities.npy'))
    test_cost=nnp.load(os.path.join(dump_dir,'test_costs.npy'))
    sig_cost=nnp.load(os.path.join(dump_dir,'sig_costs.npy'))
else:
    for data,label in tqdm(test_loader):
        #output=circuit(weights,inputs=data)
        cost,fid=semi_classical_cost(weights,inputs=data,quantum_cost=circuit,return_fid=True)
        test_fid.append(fid)
        test_cost.append(cost)
    for data,label in tqdm(sig_loader):
        cost,fid=semi_classical_cost(weights,inputs=data,quantum_cost=circuit,return_fid=True)
        sig_fid.append(fid)    
        sig_cost.append(cost)
    sig_fid=nnp.concatenate(sig_fid,axis=0)
    test_fid=nnp.concatenate(test_fid,axis=0)
    test_cost=nnp.concatenate(test_cost,axis=0)
    sig_cost=nnp.concatenate(sig_cost,axis=0)


if args.dump:
    nnp.save(os.path.join(dump_dir,'test_fidelities.npy'),test_fid)
    nnp.save(os.path.join(dump_dir,'sig_fidelities.npy'),sig_fid)
    nnp.save(os.path.join(dump_dir,'test_costs.npy'),test_cost)
    nnp.save(os.path.join(dump_dir,'sig_costs.npy'),sig_cost)

qcd_labels=nnp.zeros_like(test_fid)
sig_labels=nnp.ones_like(sig_fid)
labels=nnp.concatenate([nnp.zeros_like(test_fid),nnp.ones_like(sig_fid)],axis=0)
fids=nnp.concatenate([test_fid,sig_fid],axis=0)
costs=nnp.concatenate([test_cost,sig_cost],axis=0) 
scaler = MinMaxScaler(feature_range=(0, 1.))

costs=scaler.fit_transform(costs.reshape(-1,1)).flatten()

fpr,tpr,thresholds=roc_curve(labels,costs)
roc_auc=roc_auc_score(labels,costs)


bins_qcd,edges_qcd=nnp.histogram(test_fid,bins=80,range=[96,100])
bins_sig,edges_sig=nnp.histogram(sig_fid,bins=80,range=[96,100])
plt.stairs(bins_qcd,edges_qcd,fill=True,label='QCD')
plt.stairs(bins_sig,edges_sig,fill=False,label='$M_{grav}=4.5$ TeV')
plt.minorticks_on()
plt.grid(True,which='major',linestyle='--')

plt.xlabel('Quantum Fidelity (%): $<T|R>$')
plt.ylabel('No. of events')
plt.yscale('log')
plt.legend(loc='upper left')
plt.savefig(os.path.join(plot_dir,'fidelity_hist_grav4p5.png'))

plt.clf()
plt.plot(fpr,tpr,label='AUC = %0.3f' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC: $M_{grav} = 4.5$ TeV')
plt.minorticks_on()
plt.grid(True,which='major',linestyle='--')
plt.legend(loc='lower right')
plt.savefig(os.path.join(plot_dir,'roc_curve_grav4p5.png'))

