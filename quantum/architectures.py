 # pylint: disable=maybe-no-member
from typing import Optional, Callable, Union, List, Dict, Tuple, Any
import pennylane as qml
from helpers.utils import getIndex
from itertools import combinations
import time
from tqdm import tqdm
import pennylane.numpy as np
import os,pathlib
import helpers.utils as ut
import subprocess
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
# Global variable initialization
dev = None
all_wires=None
two_comb_wires = None

auto_wires = None
ref_wires = None

num_layers = None
index = None
n_trash_qubits = -1
SEPARATE_ANCILLA=False

params_per_wire = None
def sigmoid(x):
    return 1/(1+np.exp(-x))

def initialize(wires:int=4,layers:int=1,params:int=3):
    """
    Initializes the wire(qubit) indices, creates the two-combinations and sets up other necessary variables globally

    Args:
        wires (int): Number of wires (qubits) for the circuit.
    """

    global all_wires, auto_wires, two_comb_wires, index,num_layers,params_per_wire
    N_QUBITS=wires
    all_wires=[_ for _ in range(N_QUBITS)]
    params_per_wire=params
    two_comb_wires=list(combinations([i for i in range(wires)],2))
    auto_wires=all_wires[:wires]
    num_layers=layers
    index={'eta':getIndex('particle','eta'),'phi':getIndex('particle','phi'),'pt':getIndex('particle','pt')}

def set_device(shots:int=5000,device_name:str='default.qubit')-> qml.Device:
    """
    Sets the device on which to simulate/run the quantum circuit

    Args:
        shots (int): Number of shots for each measurement of an observable.
        device_name (str): Name of the quantum device to use.

    Returns:
        qml.Device: Initialized Pennylane device.
    """
    global dev
    dev=qml.device(device_name,wires=len(all_wires),shots=shots)
    print(dev)
    return dev
def print_training_params()->None:
    """
    Prints out the initialized training parameters for sanity check.
    Pauses for a short time to allow the user to review the parameters.
    """
    print("\n Sanity check: \n")
    print('all_wires:',all_wires)
    print()
    print('auto_wires:',auto_wires)
    print('two_comb_wires:',two_comb_wires)
    print('no. of layers:',num_layers)
    print('index:',index)
    print('trainable parameters per qumode: ',params_per_wire)
    print("\n ############################################## \n")
    print("Sleep on it for 3s")
    print("Maybe you want to change something?")
    print("Then press CTRL-C")
    print("\n ############################################## \n")
    time.sleep(3)
    print("LETS GOOOOOOOOOOOOO")
    time.sleep(1)

def circuit(weights: np.ndarray, inputs: np.ndarray) -> float:
    """
    Defines the CV quantum circuit using Strawberry Fields Fock device.
    
    Args:
        weights (np.ndarray): Trainable parameters for rotations and squeezing.
        inputs (np.ndarray): Input data with shape (N, 3) for N qumodes, where each has [pt, eta, phi].
    
    Returns:
        float: Mean photon number across all qumodes.
    """
    N = len(auto_wires)  # Number of qumodes
    sf=10*sigmoid(weights[-3])+0.01
    # State preparation for each qumode
    for w in auto_wires:
        eta = np.squeeze(inputs[:,w, index['eta']]) # corresponding to eta
        phi = np.squeeze(inputs[:,w, index['phi']]) # corresponding to phi
        pt = np.squeeze(inputs[:,w, index['pt']]) # corresponding to pt
        if inputs.shape[0]==1:
            eta=eta.item()
            phi=phi.item()
            pt=pt.item()
        qml.Displacement(sf*pt, eta, wires=w)
        qml.Squeezing(eta, pt*phi/2., wires=w)
        
    # Apply layers
    
    for L in range(num_layers):
        # Entangle qumode pairs
        for pair in two_comb_wires:
            qml.ControlledAddition(1.0, wires=pair)  
            #qml.Beamsplitter(np.pi/4.,np.pi/2., wires=[w, (w+1)%N])  # ring of CXs
        
        # Parameterized rotations and squeezing
        start = 3 * L * N
        phi_params = weights[start : start + N]
        theta_params = weights[start + N : start + 2 * N]
        omega_params = weights[start + 2 * N : start + 3 * N]
        
        for w in auto_wires:
            phi = phi_params[w]
            theta = theta_params[w]
            omega = omega_params[w]
            #qml.Beamsplitter(theta,phi, wires=[w, (w+1)%N])  # ring of BS (literally xD)
            qml.Displacement(theta,phi, wires=w)
            qml.Squeezing(omega, np.pi/4, wires=w)
                        
    return [qml.expval(qml.NumberOperator(wires=w)) for w in range(3)]#qml.expval(qml.NumberOperator(0))
    
    

def VQC_circuit(weights: np.ndarray, inputs: Optional[np.ndarray] = None) -> Any:
    """
    Defines the quantum autoencoder (QAE) circuit using provided weights and inputs.

    Args:
        weights (np.ndarray): Circuit parameters (AKA weights) for rotations.
        inputs (np.ndarray): Input data to be used in the circuit. Defaults to None.

    Returns:
        Any: Expected value of Pauli-Z tensor product on the ancillary qubits.
    """
    # State preparation for all wires
    N = len(auto_wires)  # Assuming wires is a list like [0, 1, ..., N-1]
    # State preparation for all wires
    sf=2*np.pi*sigmoid(weights[-2])+1
    #sf=0.5+sigmoid(weights[-2])
    #sfb=weights[-3]
    # for w in auto_wires:
    #     qml.PauliX(wires=w)
    
        # QAE Circuit
    for L in range(num_layers):
        for w in auto_wires:
        # Variables named according to spherical coordinate system, it's easier to understand :)
            zenith = np.squeeze(inputs[:,w, index['eta']]) # corresponding to eta
            azimuth = np.squeeze(inputs[:,w, index['phi']]) # corresponding to phi
            radius = np.squeeze(inputs[:,w, index['pt']]) # corresponding to pt
            if inputs.shape[0]==1:
                zenith=zenith.item()
                azimuth=azimuth.item()
                radius=radius.item()
            qml.RY(sf*radius*zenith, wires=w)
            qml.RX(sf*radius*azimuth, wires=w)
            
        start=3*L*N
        
        for w in auto_wires:
            qml.CNOT(wires=[w,(w+1)%N])  # ring of CNOTs

        for phi,theta,omega,w in zip(weights[start:start+N],weights[start+N:start+2*N],weights[start+2*N:start+3*N],auto_wires):
            qml.Rot(0.,theta,omega,wires=w) # perform arbitrary rotation in 3D space instead of RX/RY rotation
            qml.RX(phi,wires=w) # perform arbitrary rotation in 3D space instead of RX/RY rotation
            
    return qml.expval(qml.PauliZ(0))  


def QCNN(weights: np.ndarray, inputs: Optional[np.ndarray] = None) -> Any:
    """
    Defines the quantum autoencoder (QAE) circuit using provided weights and inputs.

    Args:
        weights (np.ndarray): Circuit parameters (AKA weights) for rotations.
        inputs (np.ndarray): Input data to be used in the circuit. Defaults to None.

    Returns:
        Any: Expected value of Pauli-Z tensor product on the ancillary qubits.
    """
    # State preparation for all wires
    N = len(auto_wires)  # Assuming wires is a list like [0, 1, ..., N-1]
    # State preparation for all wires
    sf=2*np.pi*sigmoid(weights[-2])+1
    #sfb=weights[-3]
    # for w in auto_wires:
    #     qml.PauliX(wires=w)
    
        # QAE Circuit
    for w in auto_wires:
    # Variables named according to spherical coordinate system, it's easier to understand :)
        zenith = np.squeeze(inputs[:,w, index['eta']]) # corresponding to eta
        azimuth = np.squeeze(inputs[:,w, index['phi']]) # corresponding to phi
        radius = np.squeeze(inputs[:,w, index['pt']]) # corresponding to pt
        if inputs.shape[0]==1:
            zenith=zenith.item()
            azimuth=azimuth.item()
            radius=radius.item()
        qml.RY(sf*radius*zenith, wires=w)
        qml.RZ(sf*radius*azimuth, wires=w)
            
        
    for w in auto_wires:
        qml.CY(wires=[w,(w+1)%N])  # ring of CNOTs

    for L in range(num_layers):
        phi=weights[2*L]
        theta=weights[2*L+1]
        #omega=weights[3*L+2]
        for w in auto_wires[:-(1+L)]:
            
            qml.RZ(phi,wires=w) # perform arbitrary rotation in 3D space instead of RX/RY rotation
            qml.RY(theta,wires=w) # perform arbitrary rotation in 3D space instead of RX/RY rotation
            qml.RZ(phi,wires=(w+L+1)%N) # perform arbitrary rotation in 3D space instead of RX/RY rotation
            qml.RY(theta,wires=(w+L+1)%N) # perform arbitrary rotation in 3D space instead of RX/RY rotation
            qml.CNOT(wires=[w,(w+1+L)%N])  # ring of CNOTs
            
    return [qml.expval(qml.PauliZ(i)) for i in auto_wires]  




class QuantumClassifier:
    """
    A class that constructs a Quantum Autoencoder (QAE) using pre-defined circuits.

    Args:
        wires (int): Number of qubits to use.
        shots (int): Number of shots for measurements on quantum states.
        trash_qubits (int): Number of trash qubits used in the circuit.
        dev_name (str): Name of the quantum device to use.
        backend_name (str): Backend for the QNode (e.g., 'autograd', 'torch', 'jax').
        test (bool): If True, sets the circuit immediately for testing.
    """
    def __init__(self, wires:int=4,shots=5000,dev_name:str='default.qubit',\
            ancilla:bool=False,backend_name:str='autograd',layers:int=1,test=False,params:int=3):
        initialize(wires=wires,layers=layers,params=params)
        self.device=set_device(shots=shots,device_name=dev_name)
        self.backend=backend_name
        self.current_weights=None
        self.circuit = None
        self.ancilla = ancilla
        if test: self.set_circuit() # Set the circuit for inference
    def set_circuit(self,circuit_type='normal')->None:
        """
        Configures the QNode circuit for the quantum autoencoder.

        """
        if circuit_type=='CNN':
            print("Using CNN circuit with no. of conv layers = no. of qubits (for now)")
            print("Things might be slow")
            time.sleep(2)
            self.circuit=qml.QNode(QCNN,self.device,interface=self.backend)
        else:
            self.circuit = qml.QNode(circuit,self.device,interface=self.backend)
    def fetch_circuit(self) -> qml.QNode:
        """
        Retrieves the quantum circuit for inference or training.

        Returns:
            qml.QNode: Configured quantum node (QNode) circuit.
        """
        if self.circuit is None:
            self.set_circuit()
        return self.circuit
    def fetch_backend(self) -> str:
        """
        Fetches the backend being used for the QNode.

        Returns:
            str: Name of the backend (e.g., 'autograd', 'torch').
        """
        return self.backend
    def load_weights(self,model_path:str,train:bool=False):
        """
        Loads the pre-trained weights for the autoencoder from the given file.

        Args:
            model_path (str): Path to the file containing the pre-trained model.
            train (bool): If True, enables gradients for the weights.
        """
        dictionary=ut.Unpickle(model_path)
        self.current_weights=np.array(dictionary['weights'],requires_grad=train)
    def print_weights(self):
        """
        Prints the current weights of the quantum autoencoder.
        """
        print('Current weights: \n\n',self.current_weights)

    def run_inference(self,data:np.ndarray=None,labels:np.ndarray=None,loss_fn:Callable=None,loss_type='BCE'):
        """
        Runs inference on the autoencoder circuit using the loaded weights.

        Args:
            data (np.ndarray): Input data for the quantum circuit.
            labels (np.ndarray): Truth Labels for the input data.
            loss_fn (Callable): Loss function to calculate the quantum cost.

        Returns:
            Tuple[float, float]: Inference MSE loss and classifier scores.
        """

        #print("Running in inference mode \n No batching will be performed so don't expect a progress bar")
        if self.current_weights is None:
            raise ValueError('Weights not initialized. Load a model first by calling load_weights(model_path)')       
        costs,scores=loss_fn(self.current_weights,inputs=data,labels=labels,quantum_circuit=self.circuit,return_scores=True,loss_type=loss_type)
        print("Done")
        return costs,scores
    
class QuantumTrainer():
    """
    A class for training a given quantum circuit.

    Args:
        model (QuantumClassifier): The quantum autoencoder model to be trained.
        lr (float): Learning rate for the optimizer.
        optimizer (Callable): The optimizer used for training.
        loss_fn (Callable): The loss function used for optimization.
        save (bool): Whether to save the trained model and checkpoints.
        train_max_n (int): Maximum number of training samples.
        valid_max_n (int): Maximum number of validation samples.
        epochs (int): Number of training epochs.
        patience (int): Patience for early stopping.
        kwargs (dict): Additional keyword arguments for training.
    """
    def __init__(self, model: QuantumClassifier, lr: float = 0.001, optimizer: Callable = None, 
                 loss_fn: Callable = None, save: bool = True, train_max_n: int = 100000, 
                 valid_max_n: int = 20000, epochs: int = 20, patience: int = 2, \
                    improv:float=0.01,wandb=None, lr_decay:bool=False,loss_type='BCE',**kwargs: Any) -> None:
        self.circuit=model.fetch_circuit()
        self.backend=model.fetch_backend()
        self.init_weights=kwargs['init_weights']
        self.batch_size=kwargs['batch_size'] or 1000
        self.logger=kwargs['logger']
        self.train_max_n=train_max_n
        self.valid_max_n=valid_max_n
        self.lr_decay=lr_decay
        self.epochs=epochs
        self.patience=patience
        self.saving=save
        self.current_weights=self.init_weights
        self.optim=optimizer
        self.quantum_loss=loss_fn
        self.loss_type=loss_type
        self.current_epoch=0
        self.improv=improv
        self.is_evictable=False
        self.wandb=wandb
        self.history={'train':[],'val':[],'auc':[]}
        print (f'Performing optimization with: {self.optim} | Setting Learning rate: {lr}')
        print ('Backend:',self.backend,'\n')

    def iteration(self,data: np.ndarray, labels: np.ndarray, train: bool = False) -> Union[float, Tuple[float, float]]:
        """
        Performs a single training or validation iteration.

        Args:
            data (np.ndarray): Batch of input data.
            train (bool): Whether to perform training (True) or validation (False).

        Returns:
            Union[float, Tuple[float, float]]: Training loss (or validation loss and fidelity).
        """
        if train:
            self.current_weights, cost = self.optim.step_and_cost(self.quantum_loss,self.current_weights,\
                                                                  inputs=data, labels=labels, quantum_circuit=self.circuit,loss_type=self.loss_type)
            #print(grads.shape)
            return float(cost)
        else: 
            cost,scores=self.quantum_loss(self.current_weights,inputs=data, labels=labels, \
                                          quantum_circuit=self.circuit,return_scores=True,loss_type=self.loss_type)
            
            return float(cost),float(scores)
    def is_evictable_job(self,seed:bool=None):
        """
        Marks the current job as evictable and enables copying of checkpoints to EOS.

        Args:
            seed (int, optional): Random seed for checkpoint saving. Defaults to None.
        """
        self.is_evictable=True
        self.seed=seed
    
    def run_training_loop(self,train_loader:DataLoader,val_loader:DataLoader):
        """
        Executes the full training loop, including training and validation.

        Args:
            train_loader (Any): DataLoader for the training dataset.
            val_loader (Any): DataLoader for the validation dataset.

        Returns:
            Dict[str, List[float]]: Training and validation history (losses and accuracies).
        """
        self.print_params('Initial weights: ')
        n_decays=0
        last_decay=0
        COMPLETE=False
        for n_epoch in tqdm(range(self.epochs+1)):
            sample_counter=0
            batch_yield=0
            self.current_epoch=n_epoch
            
            losses=0.
            if (n_epoch>4):
                recent_val_metrics = self.history['auc'][-2:]
                previous_val_metric = self.history['auc'][-3]
                improvement = (np.mean(recent_val_metrics) - previous_val_metric)

                if improvement < self.improv:
                    # Handle learning rate decay
                    if self.lr_decay:
                        if (n_decays < self.patience) and ((n_epoch - last_decay) >= 2):
                            last_decay = self.current_epoch
                            n_decays += 1
                            self.optim.stepsize *= 0.5  # Use a parameter if needed
                            self.logger.info(f'No improvement observed over last 3 epochs. \n Learning rate decayed to {self.optim.stepsize} at epoch {n_epoch}')
                        elif n_decays >= self.patience:
                            self.logger.info(f"\n\n No improvement over last 3 epochs and {self.patience} decay steps. Early stopping! \n\n")
                            self.save(self.save_dir, name='trained_model.pickle')
                            COMPLETE=True
                            break
                    else:
                        # Early stopping without decay
                        self.logger.info(f"\n\n No improvement over last 3 epochs. Early stopping! \n\n")
                        self.save(self.save_dir, name='trained_model.pickle')
                        COMPLETE=True
                        break
                    
            if n_epoch>0:
                print("Start Training")  
                start=round(time.time(),2)
                
                for data,labels in tqdm(train_loader,total=int(self.train_max_n/self.batch_size)):
                    sample_counter+=data.shape[0]
                    batch_yield+=1
                    loss=self.iteration(data,labels=labels,train=True)
                    losses+=loss
                    if self.wandb is not None:
                        self.wandb.log({'train_loss': losses/batch_yield})
                end=round(time.time(),2)
                train_loss=losses/batch_yield
                self.print_params('Current weights: \n\n')
                print ('Now validating!')
            else:
                print ('Running initial validation pass')
            ### Validation pass ###
            val_loss=0.
            val_batch_yield=0
            val_score=[]
            val_labels=[]
            for data,labels in tqdm(val_loader,total=int(self.valid_max_n/self.batch_size)):
                loss,score=self.iteration(data,labels=labels,train=False)
                val_loss+=loss
                val_score.append(score)
                val_labels.append(labels)
                val_batch_yield+=1
            val_loss=val_loss/val_batch_yield
            val_labels=np.array(val_labels).flatten()
            val_score=np.array(val_score).flatten()
            val_auc=roc_auc_score(val_labels,val_score)
            print("Val shape:", val_score.shape)
            val_std=np.std(val_score)
            val_score=np.mean(val_score)
            if self.wandb is not None:
                self.wandb.log({'val_loss': val_loss})
                self.wandb.log({'val_auc': val_auc})
            #print (f'Epoch {n_epoch}: Train Loss:{train_loss} Val loss: {val_loss}')
            if n_epoch>0:
                self.logger.info(f'Epoch {n_epoch}: Network with {len(auto_wires)} input qubits trained on {sample_counter} samples in {batch_yield} batches')
                self.logger.info(f'Epoch {n_epoch}: Train Loss = {train_loss:.3f} | Val loss = {val_loss:.3f} | Val preds mean and std = {val_score:.3f} , {val_std:.3f}\
                | Val AUC = {val_auc:.3f} \n Time taken = {end-start:.3f} seconds \n\n')
                
                self.history['train'].append(train_loss)

            else:
                self.logger.info(f'Initial validation pass completed')
                self.logger.info(f'Epoch {n_epoch} (No training performed): Val loss = {val_loss:.3f} | Val AUC = {val_auc:.3f} | Val preds mean and std = {val_score:.3f} , {val_std:.3f}\n\n')
            self.history['val'].append(val_loss)
            self.history['auc'].append(val_auc)
            if self.saving:
                if (n_epoch==self.epochs):
                    name='trained_model.pickle'
                elif n_epoch==0:
                    name='init_weights.pickle'
                else:
                    name=None
                self.save(self.save_dir,name=name)
                if (self.is_evictable)&(n_epoch>0):
                    print ('Will copy over checkpoints')
                    name='ep{:02}.pickle'.format(self.current_epoch)
                    try:
                        # Fetch the seed by splitting the save_dir - last directory in tree should be the seed
                        tmpfile=f"{os.environ['EOS_MGM_URL']}://eos/user/{os.environ['CERN_USERNAME'][0]}/{os.environ['CERN_USERNAME']}/QML/checkpoint_dumps/{self.seed}/{name}"
                        exec=os.path.join(os.environ['BELLE2_EXEC'],'xrdcp')
                        subprocess.call(f'{exec} {os.path.join(self.checkpoint_dir,name)} {tmpfile}',shell=True)
                    except:
                        print("Failed to copy over checkpoints")
                        pass
        if not COMPLETE:
            # pickle the history
            ut.Pickle(self.history,'history.pickle',path=self.save_dir)
        return self.history
    
    def print_params(self,prefix: Optional[str]=None) -> None:
        """
        Prints the current parameters (weights) of the quantum autoencoder.

        Args:
            prefix (Optional[str]): An optional prefix to print before the parameters. 
                                    Defaults to None.
        """
        if prefix is not None: print (prefix)
        print('autograd weights:',self.current_weights,'\n')
        
    def save(self, save_dir: str, name: Optional[str] = None) -> None:
        """
        Saves the model weights to a specified directory.

        Args:
            save_dir (str): Directory where the model weights will be saved.
            name (Optional[str]): The name of the file to save. Defaults to None.
                                If not provided, the file name will be based on the current epoch.
        """
        opt_name=None
        if name is None:
            if self.current_epoch>100: 
                name = 'ep{:03}.pickle'.format(self.current_epoch)
                opt_name='optimizer_ep{:03}.json'.format(self.current_epoch)
            else:
                name='ep{:02}.pickle'.format(self.current_epoch)
                opt_name='optimizer_ep{:02}.json'.format(self.current_epoch)
        if 'trained' not in name: save_dir=self.checkpoint_dir
        else:
            opt_name='optimizer.json'
        ut.Pickle({'weights':self.current_weights},name,path=save_dir)
        try:
            optim_dict={'stepsize':self.optim.stepsize,'beta1':self.optim.beta1,'beta2':self.optim.beta2,'epsilon':self.optim.eps,\
                        'fm':self.optim.fm,'sm':self.optim.sm,'t':self.optim.t}
            # save dict to json file
            import json
            with open(os.path.join(save_dir,opt_name), 'w') as f:
                json.dump(optim_dict, f)
            print("Optimizer state saved")
        except Exception as e:
            print(f"Error saving optimizer state: {str(e)}")

        #print(f"Model saved to {os.path.join(save_dir,name)}")
    def get_current_epoch(self) -> None:
        """
        Returns the current epoch number during training.

        Returns:
            int: The current epoch number.
        """
        return self.current_epoch
    def set_current_epoch(self,epoch:int)->None:
        """
        Sets the current epoch number if training is resumed from a checkpoint.

        Args:
            epoch (int): The epoch number to set.
        """
        self.current_epoch=epoch+1
        print("Resume training from epoch:",epoch+1)
    def set_directories(self,save_dir: str) -> None:
        """
        Sets up directories for saving model checkpoints and logs.

        Args:
            save_dir (str): The directory where model and checkpoints will be saved.
        """
        self.save_dir=save_dir
        self.checkpoint_dir=os.path.join(save_dir,'checkpoints')
        pathlib.Path(self.checkpoint_dir).mkdir(parents=True,exist_ok=True)
    
    def fetch_history(self) -> Dict[str, List[float]]:
        """
        Fetches the history of training and validation losses and accuracies.

        Returns:
            Dict[str, List[float]]: A dictionary containing the history of training and 
                                    validation losses and accuracies.
        """
        return self.history


