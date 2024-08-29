from argparse import ArgumentParser
import torch,glob,time
from torch.utils.data import DataLoader,Dataset
from itertools import combinations
import helpers.utils as ut
import pennylane as qml
from tqdm import tqdm
import matplotlib.pyplot as plt
import case_reader as cr
from utils import check_dir,print_events,Pickle,Unpickle
import os,sys,pathlib
from loguru import logger
from functools import partial
import datetime
parser=ArgumentParser(description='select options to train quantum autoencoder')
parser.add_argument('--seed',default=9999,type=int,help='Some number to index the run')
parser.add_argument('--train',default=False,action='store_true',help='train network!')
parser.add_argument('--wires',default=4,type=int,help='number of wires/qubits that the circuit needs to process(AB system)')
parser.add_argument('--trash-qubits',default=1,type=int,help='number of qubits defining the B system, or the reference and trash states!')
parser.add_argument('--shots',default=5000,type=int)
parser.add_argument('-b','--batch-size',default=1,type=int)
parser.add_argument('-e','--epochs',default=20,type=int)
parser.add_argument('--backend',default='autograd')
parser.add_argument('--lr',default=0.01,type=float)
parser.add_argument('--save',default=False,action='store_true')
parser.add_argument('--train-file',default='sample_bg_morestat.csv')
parser.add_argument('--test',default=False,action='store_true')
parser.add_argument('--path',default='',type=str)
parser.add_argument('--test-device',default='default.qubit')
args=parser.parse_args()

args.non_trash=args.wires-args.trash_qubits
assert args.non_trash>0,'Need strictly positive dimensional compressed representation of input state!'

import pennylane.numpy as np



print(f"args.wires: {args.wires}")
print(f"args.trash_qubits: {args.trash_qubits}")
if args.save:
    save_dir=os.path.join(ut.path_dict['QAE_save'],str(args.seed))
    pathlib.Path(save_dir).mkdir(parents=True,exist_ok=True)
    checkpoint_dir=os.path.join(save_dir,'checkpoints')
    pathlib.Path(checkpoint_dir).mkdir(parents=True,exist_ok=True)
    plot_dir=os.path.join(save_dir,'plots')
    pathlib.Path(plot_dir).mkdir(parents=True,exist_ok=True)
    print("Will save models to: ",save_dir)

logger.add(os.path.join(save_dir,'logs.log'),rotation='10 MB',backtrace=True,diagnose=True,level='DEBUG', mode="w")

        
if args.train:
    dev=qml.device('lightning.kokkos',wires=args.wires+args.trash_qubits+1,shots=args.shots)
    two_comb_wires=list(combinations([i for i in range(args.wires)],2))   
    all_wires=[_ for _ in range(args.wires+args.trash_qubits+1)]
    ancillary_wires=all_wires[-1:]
    print ('Ancillary wires:',ancillary_wires)
    auto_wires=all_wires[:args.wires]
    ref_wires=all_wires[args.wires:args.wires+args.trash_qubits] 
elif args.test:
    test_args=Unpickle('args',path=args.path)
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
    
index={'eta':ut.getIndex('particle','eta'),'phi':ut.getIndex('particle','phi'),'pt':ut.getIndex('particle','pt')}

if not args.test: 
    print ('Selected argument namespace: ')
    for key,item in vars(args).items():
        print (f'\t {key:12} : {repr(item):50}')
    print ('\n')


#@qml.qnode(dev,interface=args.backend)
def batch_semi_classical_cost(weights,inputs=None,quantum_cost=None,return_fid=False):
    if return_fid:
        cost,fid=[],[]
        #for item in inputs:
        #    fid.append(quantum_cost(weights,item))
        #    cost.append(1-fid[-1])
        fid=quantum_cost(weights,inputs)
        cost=1.-fid
        return np.array(cost).mean(),np.array(fid).mean()
    # the minus sign is to maximize the fidelity between the trash and reference states - which implies that the output and input states are close to each other
    batched_average_cost=np.array(1-quantum_cost(weights,inputs)).mean()#np.array([-quantum_cost(weights,item) for item in inputs]).mean()
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
        #self.optim=qml.QNGOptimizer(lr,approx='diag',lam=10e-12)# Previously had the deprecated argument diag_approx=True
        self.optim=qml.AdamOptimizer()# Previously had the deprecated argument diag_approx=True
        self.quantum_loss=batch_semi_classical_cost
        #self.setup_quantum()
        print (f'Performing quantum gradient with: {self.optim}  Learning rate: {lr}')
        print ('Backend:',self.backend,'\n')
        
    # def setup_quantum(self):
    #     # print ('Setting up quantum training procedure!')
    #     # obs=[qml.operation.Tensor(*[qml.PauliZ(i) for i in ancillary_wires])]
    #     # coeffs=[1.]
        
        
    #     # self.hamiltonian=qml.Hamiltonian(coeffs,obs)
    #     #self.quantum_cost=self.model#qml.ExpValCost(self.model,self.hamiltonian,dev,optimize=True)
    #     #self.metric_tensor=qml.metric_tensor(self.model,approx='diag')
    #     n=self.init_weights.flatten().shape[0]
    #     diag_reg=np.zeros((n,n))
    #     np.fill_diagonal(diag_reg,self.optim.lam)
    #     self.diag_reg=(diag_reg)
    #     self.loss_fn=batch_semi_classical_cost
    #@qml.qnode(dev,interface=args.backend)
    #def loss_fn(self,weights, inputs=None):
    #    return np.array([-self.model(weights,item) for item in inputs]).mean()
    
    def iteration(self,data,train=False):
        if train:
            #grad,cost=self.optim.compute_grad(self.quantum_loss,[self.current_weights],dict(inputs=data,quantum_cost=self.quantum_cost))
            #self.metric_tensor=lambda p:qml.metric_tensor(self.model,approx='diag')(p,data)
            #self.optim.metric_tensor = np.mean([self.metric_tensor(self.current_weights,inputs=item) for item in data], axis=0) + self.diag_reg
            #self.optim.metric_tensor=(np.mean([self.metric_tensor(self.current_weights,inputs=item) for item in data],axis=0)+self.diag_reg)
            #self.current_weights=self.optim.apply_grad(grad[0],self.current_weights)
            self.current_weights, cost = self.optim.step_and_cost(self.quantum_loss,self.current_weights,inputs=data,quantum_cost=self.model)
            # Update current weights
            #self.current_weights = grad
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
        Pickle({'weights':self.current_weights},name,path=save_dir)
            

@qml.qnode(dev,interface=args.backend)
def circuit(weights,inputs=None):

    # State preparation for all wires
    N = len(auto_wires)  # Assuming wires is a list like [0, 1, ..., N-1]
    # State preparation for all wires
    for w in auto_wires:
        # Variables named according to spherical coordinate system, it's easier to understand :)
        #import pdb;pdb.set_trace()
        zenith = inputs[:,w, index['eta']] # corresponding to eta
        azimuth = inputs[:,w, index['phi']] # corresponding to phi
        radius = inputs[:,w, index['pt']] # corresponding to pt
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
    for ref_wire,trash_wire in zip(ref_wires,auto_wires[-args.trash_qubits:]):
        qml.CSWAP(wires=[ancillary_wires[0], ref_wire, trash_wire])
    qml.Hadamard(ancillary_wires)
    return qml.expval(qml.operation.Tensor(*[qml.PauliZ(i) for i in ancillary_wires]))
    
    
# def autoenc(weights,inputs=None):    
#     circuit(weights,inputs=inputs)
#     obs=[qml.operation.Tensor(*[qml.PauliZ(i) for i in ancillary_wires])]
#     coeffs=np.array([1.])
#     hamiltonian=qml.Hamiltonian(coeffs,obs)
#     # returns the expected fidelity between the reference state and the trash state
#     return qml.expval(hamiltonian)
    #return qml.expval(qml.operation.Tensor(*[qml.PauliZ(i) for i in ancillary_wires]))
if args.train:
    init_weights=np.random.uniform(0,np.pi,size=(len(auto_wires),), requires_grad=True)
    if args.save:
        Pickle(args,'args',path=save_dir)
        with open(os.path.join(save_dir,'args.txt'),'w+') as f:
            f.write(repr(args))
    train_filelist=sorted(glob.glob(ut.path_dict['QCD_train']+'/*.h5'))
    val_filelist=sorted(glob.glob(ut.path_dict['QCD_test']+'/*.h5'))
    train_loader = cr.CASEDelphesDataLoader(filelist=train_filelist,batch_size=args.batch_size,input_shape=(len(auto_wires),3),train=True,max_samples=2e5)
    val_loader = cr.CASEDelphesDataLoader(filelist=val_filelist,batch_size=args.batch_size,input_shape=(len(auto_wires),3),train=False,max_samples=1e4) 
    
    
    
    trainer=Trainer(circuit,lr=args.lr,backend=args.backend,init_weights=init_weights)
    trainer.print_params('Initialized parameters!')
    #import pdb;pdb.set_trace()
    
    history={'train':[],'val':[],'accuracy':[]}
    first_draw=True
    logger.debug(f"Training started at {datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}")
    abs_start=time.time()
    try:
          
        for n_epoch in range(args.epochs+1):
            sample_counter=0
            batch_yield=0
            #if n_epoch !=0:
            losses=0.
            
            if n_epoch>0:
                print("Start Training")  
                start=round(time.time(),2)
                
                for data in tqdm(train_loader):
                    #print (data)
                    sample_counter+=data.shape[0]
                    batch_yield+=1
                    loss=trainer.iteration(data,train=True)
                    losses+=loss
                end=round(time.time(),2)
                train_loss=losses/batch_yield
                #else: train_loss=float('NaN')
            
                print ('Validating!')
            
            else:
                print ('Running initial validation pass')
            val_loss=0.
            val_batch_yield=0
            for data,label in tqdm(val_loader):
                loss,batch_fid=trainer.iteration(data,train=False)
                val_loss+=loss
                val_batch_yield+=1
            val_loss=val_loss/val_batch_yield
            #print (f'Epoch {n_epoch}: Train Loss:{train_loss} Val loss: {val_loss}')
            if n_epoch>0:
                logger.debug(f'Epoch {n_epoch}: Network with {len(auto_wires)} input qubits trained on {sample_counter} samples in {batch_yield} batches')
                logger.debug(f'Epoch {n_epoch}: Train Loss = {train_loss:.3f} | Val loss = {val_loss:.3f} \n Time taken = {end-start:.1f} seconds \n\n')
                trainer.print_params('Current parameters!')
                history['train'].append(train_loss)

            else:
                logger.debug(f'Initial validation pass completed')
                logger.debug(f'Epoch {n_epoch} (No training performed): Val loss = {val_loss:.3f} \n\n')
            history['val'].append(val_loss)
            if args.save:
                if n_epoch==args.epochs:
                    name='trained_model.pickle'
                elif n_epoch==0:
                    name='init_weights.pickle'
                else:
                    name=None
                trainer.save(n_epoch,save_dir,name=name)
    except KeyboardInterrupt:
        pass
    finally:
        logger.debug('Training completed with the following parameters:')
        logger.debug(f'Epochs: {n_epoch} | Learning rate: {args.lr} | Batch size: {args.batch_size} \nBackend: {args.backend} | Wires: {args.wires} | Trash qubits: {args.trash_qubits} | Shots: {args.shots} \n')
        trainer.print_params('Trained parameters:')
        print (history)
        if args.save:
            done_epochs=len(history['train'])
            Pickle(history,'history',path=save_dir)
            fig,axes=plt.subplots(figsize=(15,12))
            axes.plot(np.arange(done_epochs),history['train'],label='train',linewidth=2)
            axes.plot(np.arange(done_epochs+1),history['val'],label='val',linewidth=2)
            #if len(history['accuracy'])==done_epochs+1:
            #    axes.plot(np.arange(done_epochs),history['accuracy'],label='val accuracy',linewidth=2)
            axes.set_xlabel('Epochs',size=25)
            axes.set_ylabel('$1-<T|F>$',size=25)
            axes.set_xticks(np.arange(0,done_epochs+1,5))
            axes.legend(prop={'size':25})
    
            axes.tick_params(labelsize=20)
            fig.savefig(os.path.join(save_dir,'history'))
            abs_end=time.time()
            logger.debug(f"Training finished at {datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}")
            logger.debug(f"Total time taken including all overheads: {abs_end-abs_start:.2f} seconds")
    sys.exit()

if args.test:
    curr_args=args
    print ('Testing with model:',args.path)
    history=Unpickle('history',path=args.path)
    #print (history)
    val_loss=history['val']
    epoch=np.argmin(val_loss)
    print (epoch,val_loss[epoch])
    if epoch<10:
        filename='0'+str(epoch)
    else: filename=str(epoch)
    model_path=os.path.join(args.path,'ep'+filename+'.pickle')
    print ('Model path:',model_path)
    path=args.path
    args=Unpickle('args',path=args.path)
    print ('Loaded args:',args)

    dictionary=Unpickle(model_path)
    weights=dictionary['weights']

    # import numpy as nnp
    #     #init_weights=weights
    # all_data=get_data(train=False,combined_signal=False,keys=args.keys,
    #                   train_file=args.train_file,return_splitted=True,scale=True,bg_only=True)
    

    # store_dict={}
    # i=0
        
    # for key,val in all_data.items():
    #     if type(val)!= np.ndarray:
    #         store_dict[key]=val
    #         continue
       
        
    
    #     fid=[]
    #     y=[]
    #     print ('Testing: ',key,val.shape)
    #     for item in tqdm(val):
    #         fid.append(autoenc(weights,item))
            
    #     fid=nnp.array(fid)
    #     print (fid.shape,type(fid))
    #     store_dict[key]=fid
    #     i+=1
    # print_events(store_dict,name='store dict')
    
    # Pickle(store_dict,'test' if curr_args.test_device=='default.qubit' else 'test_'+'_'.join(curr_args.test_device.split('.')),path=path)
    sys.exit()
        



