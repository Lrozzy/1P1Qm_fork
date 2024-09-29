from argparse import ArgumentParser
import torch,glob,time
from torch.utils.data import DataLoader,Dataset
from itertools import combinations
import helpers.utils as ut
import helpers.path_setter as ps
import pennylane as qml
from tqdm import tqdm
import matplotlib.pyplot as plt
import case_reader as cr
import os,sys,pathlib
from loguru import logger

import datetime
parser=ArgumentParser(description='select options to train quantum autoencoder')
parser.add_argument('--seed',default=9999,type=int,help='Some number to index the run')
parser.add_argument('--train',default=False,action='store_true',help='train network!')
parser.add_argument('--wires',default=4,type=int,help='number of wires/qubits that the circuit needs to process(AB system)')
parser.add_argument('--trash-qubits',default=1,type=int,help='number of qubits defining the B system, or the reference and trash states!')
parser.add_argument('--shots',default=5000,type=int)
parser.add_argument('--train_n',default=100000,type=int)
parser.add_argument('--valid_n',default=20000,type=int)

parser.add_argument('-b','--batch-size',default=1,type=int)
parser.add_argument('-e','--epochs',default=20,type=int)
parser.add_argument('--backend',default='autograd')
parser.add_argument('--lr',default=0.01,type=float)
parser.add_argument('--save',default=False,action='store_true')
parser.add_argument('--train-file',default='sample_bg_morestat.csv')
parser.add_argument('--test',default=False,action='store_true')
parser.add_argument('--path',default='',type=str)
parser.add_argument('--device',default='default.qubit')
parser.add_argument('--desc',default='Training run')
parser.add_argument('--n_threads',type=str,default='1')
args=parser.parse_args()

args.non_trash=args.wires-args.trash_qubits
assert args.non_trash>0,'Need strictly positive dimensional compressed representation of input state!'

import pennylane.numpy as np

train_max_n=args.train_n
valid_max_n=args.valid_n

print(f"args.wires: {args.wires}")
print(f"args.trash_qubits: {args.trash_qubits}")
if args.save:
    save_dir=os.path.join(ps.path_dict['QAE_save'],str(args.seed))
    pathlib.Path(save_dir).mkdir(parents=True,exist_ok=True)
    checkpoint_dir=os.path.join(save_dir,'checkpoints')
    pathlib.Path(checkpoint_dir).mkdir(parents=True,exist_ok=True)
    plot_dir=os.path.join(save_dir,'plots')
    pathlib.Path(plot_dir).mkdir(parents=True,exist_ok=True)
    print("Will save models to: ",save_dir)

logger.add(os.path.join(save_dir,'logs.log'),rotation='10 MB',backtrace=True,diagnose=True,level='DEBUG', mode="w")

# if (args.device=='lightning.kokkos'):
#     if ('OMP_NUM_THREADS' not in os.environ.keys()):
#         os.environ['OMP_NUM_THREADS']=str(args.n_threads)
#         os.environ['OMP_PROC_BIND']='true'
if (args.device=='lightning.kokkos'):
    print(f"Initialized device {args.device} with {os.environ['OMP_NUM_THREADS']} threads")
    print(f"OMP_PROC_BIND value is : {os.environ['OMP_PROC_BIND']}")
if (args.device=='lightning.gpu'):
    print("Using GPU backend")

if args.train:
    NUM_QUBITS=args.wires+args.trash_qubits*2
    dev=qml.device(args.device,wires=NUM_QUBITS,shots=args.shots)
    two_comb_wires=list(combinations([i for i in range(args.wires)],2))   
    #all_wires=[_ for _ in range(args.wires+args.trash_qubits+1)]
    #ancillary_wires=all_wires[-1:]
    all_wires=[_ for _ in range(args.wires+args.trash_qubits*2)]
    ancillary_wires=all_wires[args.wires+args.trash_qubits:]
    print ('Ancillary wires:',ancillary_wires)
    auto_wires=all_wires[:args.wires]
    ref_wires=all_wires[args.wires:args.wires+args.trash_qubits] 
elif args.test:
    test_args=ut.Unpickle('args',path=args.path)
    dev=qml.device(args.test_device,wires=test_args.wires+test_args.trash_qubits+1,shots=test_args.shots)
    two_comb_wires=list(combinations([i for i in range(test_args.wires)],2))
    all_wires=[_ for _ in range(test_args.wires+test_args.trash_qubits+1)]
    ancillary_wires=all_wires[-1:]
    auto_wires=all_wires[:test_args.wires]
    ref_wires=all_wires[test_args.wires:test_args.wires+test_args.trash_qubits]
    del test_args
else:
    print ('Select either: --train or --test')
    sys.exit()
    
index={'eta':ps.getIndex('particle','eta'),'phi':ps.getIndex('particle','phi'),'pt':ps.getIndex('particle','pt')}

if not args.test: 
    print ('Selected argument namespace: ')
    for key,item in vars(args).items():
        print (f'\t {key:12} : {repr(item):50}')
    print ('\n')


#@qml.qnode(dev,interface=args.backend)
def batch_semi_classical_cost(weights,inputs=None,quantum_cost=None,return_fid=False):
    if return_fid:
        cost,fid=[],[]
        fid=quantum_cost(weights,inputs)
        cost=1.-fid
        return np.array(100.*cost).mean(),np.array(100.*fid).mean()
    # the minus sign is to maximize the fidelity between the trash and reference states - which implies that the output and input states are close to each other
    batched_average_cost=100.*(np.array(1-quantum_cost(weights,inputs)).mean())#np.array([-quantum_cost(weights,item) for item in inputs]).mean()
    return batched_average_cost


def collate_fn(samples):
    if type(samples[0])==tuple:
        X,y=[],[]
        for item in samples: X.append(item[0]),y.append(item[1])
        X,y=np.array(X),np.array(y).flatten()
        return X,y
    else: 
        return np.array(samples,requires_grad=False)

class Trainer(object):
    def __init__(self,model,lr=0.001,backend=None,**kwargs):
        self.model=model
        self.backend=backend
        self.init_weights=kwargs.get('init_weights')
        self.current_weights=self.init_weights
        self.metric_tensor=None
        self.optim=qml.AdamOptimizer(stepsize=lr)# Previously had the deprecated argument diag_approx=True
        self.quantum_loss=batch_semi_classical_cost
        print (f'Performing Adam optimization with: {self.optim}  Learning rate: {lr}')
        print ('Backend:',self.backend,'\n')
    
    def iteration(self,data,train=False):
        if train:
            self.current_weights, cost = self.optim.step_and_cost(self.quantum_loss,self.current_weights,inputs=data,quantum_cost=self.model)
            return float(cost)
        else: 
            cost,fid=self.quantum_loss(self.current_weights,inputs=data,quantum_cost=self.model,return_fid=True)
            return float(cost),float(fid)

    def print_params(self,prefix=None):
        if prefix is not None: print (prefix)
        print ('autograd weights:',self.current_weights,'\n')
        
    def save(self, epoch, save_dir,name=None):
        if name is None:
            if epoch>100: name = 'ep{:03}.pickle'.format(epoch)
            else: name='ep{:02}.pickle'.format(epoch)
        if 'trained' not in name: save_dir=checkpoint_dir
        ut.Pickle({'weights':self.current_weights},name,path=save_dir)


@qml.qnode(dev,interface=args.backend)
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
        
    for item in two_comb_wires: 
        qml.CNOT(wires=item)
    
    for phi,theta,omega,i in zip(weights[:N],weights[N:2*N],weights[2*N:],auto_wires):
        qml.Rot(phi,theta,omega,wires=[i]) # perform arbitrary rotation in 3D space instead of RX/RY rotation
        
    
    # SWAP test to measure fidelity
    qml.Hadamard(ancillary_wires)
    for ref_wire,trash_wire in zip(ref_wires,auto_wires[-args.trash_qubits:]):
        qml.CSWAP(wires=[ancillary_wires[0], ref_wire, trash_wire])
    qml.Hadamard(ancillary_wires)
    return qml.expval(qml.operation.Tensor(*[qml.PauliZ(i) for i in ancillary_wires]))

@qml.qnode(dev,interface=args.backend)
def reuploading_circuit(weights,inputs=None):
    # State preparation for all wires
    N = len(auto_wires)  # Assuming wires is a list like [0, 1, ..., N-1]
    # State preparation for all wires
    # Layer 1
    for w in auto_wires:
        # Variables named according to spherical coordinate system, it's easier to understand :)
        
        zenith = inputs[:,w, index['eta']] # corresponding to eta
        azimuth = inputs[:,w, index['phi']] # corresponding to phi
        radius = inputs[:,w, index['pt']] # corresponding to pt
        # Apply rotation gates modulated by the radius (pt) of the particle, which has been scaled to the range [0,1]
        qml.RY(zenith, wires=w)   
        qml.RZ(azimuth, wires=w)  
    # QAE Circuit

    for phi,theta,omega,i in zip(weights[:N],weights[N:2*N],weights[2*N:3*N],auto_wires):
        qml.Rot(phi,theta,omega,wires=[i]) # perform arbitrary rotation in 3D space instead of RX/RY rotation
    
    for item in two_comb_wires: 
        qml.CNOT(wires=item)
    
    #Layer 2
    for w in auto_wires:
        # Variables named according to spherical coordinate system, it's easier to understand :)
        
        radius = inputs[:,w, index['pt']] # corresponding to pt
        # Apply rotation gates modulated by the radius (pt) of the particle, which has been scaled to the range [0,1]
        qml.RY(radius*zenith, wires=w)   
        qml.RZ(radius*azimuth, wires=w)  
    
    # for item in two_comb_wires: 
    #     qml.CRX(radius*np.pi,wires=item)
    
    for phi,theta,omega,i in zip(weights[3*N:4*N],weights[4*N:5*N],weights[5*N:],auto_wires):
        qml.Rot(phi,theta,omega,wires=[i]) # perform arbitrary rotation in 3D space instead of RX/RY rotation
    
    for item in two_comb_wires: 
        qml.CNOT(wires=item)
    

    # SWAP test to measure fidelity
    # qml.Hadamard(ancillary_wires)
    # for ref_wire,trash_wire in zip(ref_wires,auto_wires[-args.trash_qubits:]):
    #     qml.CSWAP(wires=[ancillary_wires[0], ref_wire, trash_wire])
    # qml.Hadamard(ancillary_wires)
    
    for ref_wire,trash_wire,ancilla in zip(ref_wires,auto_wires[-args.trash_qubits:],ancillary_wires):
        qml.Hadamard(ancilla)
        qml.CSWAP(wires=[ancilla, ref_wire, trash_wire])
        qml.Hadamard(ancilla)
    
    return qml.expval(qml.operation.Tensor(*[qml.PauliZ(i) for i in ancillary_wires]))


if args.train:
    
    init_weights=np.random.uniform(0,np.pi,size=(len(auto_wires)*6,), requires_grad=True)
    if args.save:
        ut.Pickle(args,'args',path=save_dir)
        with open(os.path.join(save_dir,'args.txt'),'w+') as f:
            f.write(repr(args))
    train_filelist=sorted(glob.glob(ps.path_dict['QCD_train']+'/*.h5'))
    val_filelist=sorted(glob.glob(ps.path_dict['QCD_test']+'/*.h5'))
    train_loader = cr.CASEDelphesDataLoader(filelist=train_filelist,batch_size=args.batch_size,input_shape=(len(auto_wires),3),train=True,max_samples=train_max_n)
    val_loader = cr.CASEDelphesDataLoader(filelist=val_filelist,batch_size=args.batch_size,input_shape=(len(auto_wires),3),train=False,max_samples=valid_max_n) 
    
    
    
    trainer=Trainer(reuploading_circuit,lr=args.lr,backend=args.backend,init_weights=init_weights)
    trainer.print_params('Initialized parameters!')
    #import pdb;pdb.set_trace()
    
    history={'train':[],'val':[],'accuracy':[]}
    first_draw=True
    logger.info(f"Training started at {datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}")
    logger.info(f'Epochs: {args.epochs} | Learning rate: {args.lr} | Batch size: {args.batch_size} \nBackend: {args.backend} | Wires: {args.wires} | Trash qubits: {args.trash_qubits} | Shots: {args.shots} \n')    
    logger.info(f'Additional information: {args.desc}')
    abs_start=time.time()
    try:
        
        for n_epoch in tqdm(range(args.epochs+1)):
            sample_counter=0
            batch_yield=0
            #if n_epoch !=0:
            losses=0.
            if (n_epoch>4):
                if(abs(np.mean(history['val'][-3:])-np.mean(history['val'][-4]))<0.01):
                    print("No improvement over last 3 epochs. Early stopping!")
                    trainer.save(n_epoch,save_dir,name='trained_model.pickle')
                    break
            
            if n_epoch>0:
                print("Start Training")  
                start=round(time.time(),2)
                
                for data in tqdm(train_loader,total=2*int(train_max_n/args.batch_size)):
                    sample_counter+=data.shape[0]
                    batch_yield+=1
                    loss=trainer.iteration(data,train=True)
                    losses+=loss
                end=round(time.time(),2)
                train_loss=losses/batch_yield
                trainer.print_params('Current parameters!\n\n')
                print ('Now validating!')
            
            else:
                print ('Running initial validation pass')
            val_loss=0.
            val_batch_yield=0
            for data,label in tqdm(val_loader,total=2*int(valid_max_n/args.batch_size)):
                loss,batch_fid=trainer.iteration(data,train=False)
                val_loss+=loss
                val_batch_yield+=1
            val_loss=val_loss/val_batch_yield
            #print (f'Epoch {n_epoch}: Train Loss:{train_loss} Val loss: {val_loss}')
            if n_epoch>0:
                logger.info(f'Epoch {n_epoch}: Network with {len(auto_wires)} input qubits trained on {sample_counter} samples in {batch_yield} batches')
                logger.info(f'Epoch {n_epoch}: Train Loss = {train_loss:.3f} | Val loss = {val_loss:.3f} \n Time taken = {end-start:.3f} seconds \n\n')
                
                history['train'].append(train_loss)

            else:
                logger.info(f'Initial validation pass completed')
                logger.info(f'Epoch {n_epoch} (No training performed): Val loss = {val_loss:.3f} \n\n')
            history['val'].append(val_loss)
            if args.save:
                if (n_epoch==args.epochs):
                    name='trained_model.pickle'
                elif n_epoch==0:
                    name='init_weights.pickle'
                else:
                    name=None
                trainer.save(n_epoch,save_dir,name=name)
    except KeyboardInterrupt:
        print("WHYYYYY")
        print("DON'T PRESS CTRL+C AGAIN. I'M TRYING TO SAVE THE CURRENT MODEL AND WRITE TO LOG!") 
        trainer.save(n_epoch,save_dir,name='aborted_weights.pickle')
        trainer.print_params('Training aborted. Current parameters are: ')
    finally:
        logger.info('Training completed with the following parameters:')
        trainer.print_params('Trained parameters:')
        print (history)
        if args.save:
            done_epochs=len(history['train'])
            ut.Pickle(history,'history',path=save_dir)
            fig,axes=plt.subplots(figsize=(15,12))
            axes.plot(np.arange(done_epochs),history['train'],label='train',linewidth=2)
            axes.plot(np.arange(done_epochs+1),history['val'],label='val',linewidth=2)
            axes.set_xlabel('Epochs',size=25)
            axes.set_ylabel('$1-<T|F> $(in %)',size=25)
            axes.set_xticks(np.arange(0,done_epochs+1,5))
            axes.legend(prop={'size':25})
    
            axes.tick_params(labelsize=20)
            fig.savefig(os.path.join(save_dir,'history'))
            abs_end=time.time()
            logger.info(f"Training finished at {datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}")
            logger.info(f"Total time taken including all overheads: {abs_end-abs_start:.2f} seconds")
    

