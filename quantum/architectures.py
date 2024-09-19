from typing import Any
import pennylane as qml
from helpers.utils import getIndex
from itertools import combinations
import time
from tqdm import tqdm
import pennylane.numpy as np
import os,pathlib
import helpers.utils as ut
dev = None
all_wires=None
two_comb_wires = None
auto_wires = None
ref_wires = None
ancillary_wires = None
index = None
n_trash_qubits = -1

def initialize(wires:int=4, trash_qubits:int=0):
    global all_wires, auto_wires, two_comb_wires, ref_wires, ancillary_wires, index, n_trash_qubits
    n_trash_qubits = trash_qubits
    two_comb_wires=list(combinations([i for i in range(wires)],2))
    all_wires=[_ for _ in range(wires+trash_qubits+1)]
    ancillary_wires=all_wires[-1:]
    auto_wires=all_wires[:wires]
    ref_wires=all_wires[wires:wires+trash_qubits] # Do not initialize ref_wires before n_trash_qubits is set
    
    index={'eta':getIndex('particle','eta'),'phi':getIndex('particle','phi'),'pt':getIndex('particle','pt')}
def set_device(shots:int=5000,device_name:str='default.qubit'):
    global dev
    dev=qml.device(device_name,wires=len(all_wires),shots=shots)
    return dev
def print_training_params():
    print("\n Sanity check: \n")
    print('all_wires:',all_wires)
    print()
    print('auto_wires:',auto_wires)
    print('two_comb_wires:',two_comb_wires)
    print('ref_wires:',ref_wires)
    print('ancillary_wires:',ancillary_wires)
    print('index:',index)
    print('n_trash_qubits:',n_trash_qubits)
    print("\n ############################################## \n")

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
    for ref_wire,trash_wire in zip(ref_wires,auto_wires[-n_trash_qubits:]):
        qml.CSWAP(wires=[ancillary_wires[0], ref_wire, trash_wire])
    qml.Hadamard(ancillary_wires)
    return qml.expval(qml.operation.Tensor(*[qml.PauliZ(i) for i in ancillary_wires]))
   
class QuantumAutoencoder:
    def __init__(self, wires:int=4,trash_qubits:int=0,dev_name:str='default.qubit',backend_name:str='autograd',test=False,n_threads:str='1'):
        initialize(wires=wires,trash_qubits=trash_qubits)
        self.device=set_device(shots=5000,device_name=dev_name)
        self.backend=backend_name
        self.current_weights=None
        self.circuit = None
        if test: self.set_circuit() # Set the circuit for inference
    def set_circuit(self):
        self.circuit = qml.QNode(circuit,self.device,interface=self.backend)
    def fetch_circuit(self):
        if self.circuit is None:
            self.set_circuit()
        return self.circuit
    def fetch_backend(self):
        return self.backend
    def load_weights(self,model_path,train=False):
        dictionary=ut.Unpickle(model_path)
        self.current_weights=np.array(dictionary['weights'],requires_grad=train)
    def run_inference(self,data,loss_fn=None):
        print("Running in inference mode \n No batching will be performed so don't expect a progress bar")
        if self.current_weights is None:
            raise ValueError('Weights not initialized. Load a model first by calling load_weights(model_path)')       
        costs,fids=loss_fn(self.current_weights,inputs=data,quantum_cost=self.circuit,return_fid=True)
        print("Done")
        return costs,fids
    
class QuantumTrainer():
    def __init__(self,model:QuantumAutoencoder,lr:float=0.001,optimizer=None,loss_fn=None,save=True,train_max_n=1e5,valid_max_n=2e4,epochs=20,patience=4,**kwargs):
        self.circuit=model.fetch_circuit()
        self.backend=model.fetch_backend()
        self.init_weights=kwargs['init_weights']
        self.batch_size=kwargs['batch_size'] or 1000
        self.logger=kwargs['logger']
        self.train_max_n=2*train_max_n
        self.valid_max_n=2*valid_max_n
        self.epochs=epochs
        self.patience=patience
        self.saving=save
        self.current_weights=self.init_weights
        self.optim=optimizer# Previously had the deprecated argument diag_approx=True
        self.quantum_loss=loss_fn
        self.current_epoch=0
        self.history={'train':[],'val':[],'accuracy':[]}
        print (f'Performing optimization with: {self.optim} | Setting Learning rate: {lr}')
        print ('Backend:',self.backend,'\n')
    def iteration(self,data,train=False):
        if train:
            self.current_weights, cost = self.optim.step_and_cost(self.quantum_loss,self.current_weights,inputs=data,quantum_cost=self.circuit)
            return float(cost)
        else: 
            cost,fid=self.quantum_loss(self.current_weights,inputs=data,quantum_cost=self.circuit,return_fid=True)
            return float(cost),float(fid)
    
    def run_training_loop(self,train_loader,val_loader):
        self.print_params('Initial weights: ')
        for n_epoch in tqdm(range(self.epochs+1)):
            sample_counter=0
            batch_yield=0
            self.current_epoch=n_epoch
            
            losses=0.
            if (n_epoch>4):
                if(abs(np.mean(self.history['val'][-3:])-np.mean(self.history['val'][-4]))<0.01):
                    print("No improvement over last 3 epochs. Early stopping!")
                    self.logger.info("\n\n No improvement over last 3 epochs. Early stopping! \n\n")
                    self.save(self.save_dir,name='trained_model.pickle')
                    break
            
            if n_epoch>0:
                print("Start Training")  
                start=round(time.time(),2)
                
                for data in tqdm(train_loader,total=int(self.train_max_n/self.batch_size)):
                    sample_counter+=data.shape[0]
                    batch_yield+=1
                    loss=self.iteration(data,train=True)
                    losses+=loss
                end=round(time.time(),2)
                train_loss=losses/batch_yield
                self.print_params('Current weights: \n\n')
                print ('Now validating!')
            
            else:
                print ('Running initial validation pass')
            val_loss=0.
            val_batch_yield=0
            for data,label in tqdm(val_loader,total=int(self.valid_max_n/self.batch_size)):
                loss,batch_fid=self.iteration(data,train=False)
                val_loss+=loss
                val_batch_yield+=1
            val_loss=val_loss/val_batch_yield
            #print (f'Epoch {n_epoch}: Train Loss:{train_loss} Val loss: {val_loss}')
            if n_epoch>0:
                self.logger.info(f'Epoch {n_epoch}: Network with {len(auto_wires)} input qubits trained on {sample_counter} samples in {batch_yield} batches')
                self.logger.info(f'Epoch {n_epoch}: Train Loss = {train_loss:.3f} | Val loss = {val_loss:.3f} \n Time taken = {end-start:.3f} seconds \n\n')
                
                self.history['train'].append(train_loss)

            else:
                self.logger.info(f'Initial validation pass completed')
                self.logger.info(f'Epoch {n_epoch} (No training performed): Val loss = {val_loss:.3f} \n\n')
            self.history['val'].append(val_loss)
            if self.saving:
                if (n_epoch==self.epochs):
                    name='trained_model.pickle'
                elif n_epoch==0:
                    name='init_weights.pickle'
                else:
                    name=None
                self.save(self.save_dir,name=name)
        return self.history
    
    def print_params(self,prefix=None):
        if prefix is not None: print (prefix)
        print('autograd weights:',self.current_weights,'\n')
        
    def save(self, save_dir,name=None):
        if name is None:
            if self.current_epoch>100: name = 'ep{:03}.pickle'.format(self.current_epoch)
            else: name='ep{:02}.pickle'.format(self.current_epoch)
        if 'trained' not in name: save_dir=self.checkpoint_dir
        ut.Pickle({'weights':self.current_weights},name,path=save_dir)
    
    def get_current_epoch(self):
        return self.current_epoch
    
    def set_directories(self,save_dir):
        self.save_dir=save_dir
        self.checkpoint_dir=os.path.join(save_dir,'checkpoints')
        pathlib.Path(self.checkpoint_dir).mkdir(parents=True,exist_ok=True)
    
    def fetch_history(self):
        return self.history

