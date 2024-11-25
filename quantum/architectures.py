 # pylint: disable=maybe-no-member
from typing import Optional, Callable, Union, List, Dict, Tuple, Any
import pennylane as qml
from helpers.utils import getIndex
from itertools import combinations, product
import time
from tqdm import tqdm
import pennylane.numpy as np
import os,pathlib
import helpers.utils as ut
import subprocess
from torch.utils.data import DataLoader
# Global variable initialization
dev = None
all_wires=None
two_comb_wires = None
subjet_comb_wires = None
auto_wires = None
ref_wires = None
ancillary_wires = None
index = None
n_trash_qubits = -1
SEPARATE_ANCILLA=False

def initialize(wires:int=4, trash_qubits:int=0,separate_ancilla=False):
    """
    Initializes the wire(qubit) indices, creates the two-combinations and sets up other necessary variables globally

    Args:
        wires (int): Number of wires (qubits) for the circuit.
        trash_qubits (int): Number of qubits reserved for trash.
        separate_ancilla (bool): Whether to separate ancillary qubits from main qubits.
    """

    global all_wires, auto_wires, two_comb_wires, subjet_comb_wires, ref_wires, ancillary_wires, index, n_trash_qubits,SEPARATE_ANCILLA
    if separate_ancilla:
        SEPARATE_ANCILLA=True
        N_QUBITS=wires+trash_qubits*2
        N_ANCILLA=trash_qubits
    else:
        N_QUBITS=wires+trash_qubits+1
        N_ANCILLA=1
    all_wires=[_ for _ in range(N_QUBITS)]
    n_trash_qubits = trash_qubits
    two_comb_wires=list(combinations([i for i in range(wires)],2))
    subjet_comb_wires=list(product(all_wires[wires//2:wires],all_wires[:wires//2]))
    ancillary_wires=all_wires[-N_ANCILLA:] # The last N qubits (earlier, we had N=1) are the ancilla
    auto_wires=all_wires[:wires]
    ref_wires=all_wires[wires:wires+trash_qubits] # Do NOT initialize ref_wires before n_trash_qubits is set    
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
    print('subjet_comb_wires:',subjet_comb_wires)
    print('ref_wires:',ref_wires)
    print('ancillary_wires:',ancillary_wires)
    print('index:',index)
    print('n_trash_qubits:',n_trash_qubits)
    print("\n ############################################## \n")
    print("Sleep on it for 5s")
    print("Maybe you want to change something?")
    print("Then press CTRL-C")
    print("\n ############################################## \n")
    time.sleep(5)
    print("LETS GOOOOOOOOOOOOO")
    time.sleep(1)
def circuit(weights: np.ndarray, inputs: Optional[np.ndarray] = None) -> Any:
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
    for w in auto_wires:
        # Variables named according to spherical coordinate system, it's easier to understand :)
        
        zenith = inputs[:,w, index['eta']] # corresponding to eta
        azimuth = inputs[:,w, index['phi']] # corresponding to phi
        radius = inputs[:,w, index['pt']] # corresponding to pt
        # Apply rotation gates modulated by the radius (pt) of the particle, which has been scaled to the range [0,1]
        qml.RY(radius * zenith, wires=w)   
        qml.RZ(radius * azimuth, wires=w)  
    # QAE Circuit
    for item in two_comb_wires: 
        qml.CNOT(wires=item)

    for phi,theta,omega,i in zip(weights[:N],weights[N:2*N],weights[2*N:],auto_wires):
        qml.Rot(phi,theta,omega,wires=[i]) # perform arbitrary rotation in 3D space instead of RX/RY rotation

    # FIXED: handle the case where ancillary_wires is an integer, which is the case when separate_ancilla is False
    if len(ancillary_wires)==1:
        ancillary_wirelist=ancillary_wires*len(ref_wires)
    else:
        ancillary_wirelist=ancillary_wires 

    # SWAP test to measure fidelity
    for ref_wire,trash_wire,ancilla in zip(ref_wires,auto_wires[-n_trash_qubits:],ancillary_wirelist):
        qml.Hadamard(ancilla)
        qml.CSWAP(wires=[ancilla, ref_wire, trash_wire])
        qml.Hadamard(ancilla)
    return qml.expval(qml.operation.Tensor(*[qml.PauliZ(i) for i in ancillary_wires]))
    #fidelities = [qml.expval(qml.PauliZ(i)) for i in ancillary_wires]
    
    #return qml.sum(*fidelities) / len(ancillary_wires)
def reuploading_circuit(weights: np.ndarray, inputs: Optional[np.ndarray] = None) -> Any:
    """
    Defines the feature re-uploading quantum autoencoder (QAE) circuit with 2 layers.

    Args:
        weights (np.ndarray): Circuit parameters (AKA weights) for rotations and gates.
        inputs (np.ndarray): Input data to be used in the circuit. Defaults to None.

    Returns:
        Any: Expected value of Pauli-Z tensor product on the ancillary qubits.
    """
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
        qml.RY(radius*zenith, wires=w)   
        qml.RZ(radius*azimuth, wires=w)  
    # QAE Circuit
    for item in subjet_comb_wires:
        qml.CNOT(wires=item)
    
    for phi,theta,omega,i in zip(weights[:N],weights[N:2*N],weights[2*N:3*N],auto_wires):
        qml.Rot(phi,theta,omega,wires=[i]) # perform arbitrary rotation in 3D space instead of RX/RY rotation
    
    for item in two_comb_wires: 
        qml.CNOT(wires=item)
    
    # Layer 2
    for w in auto_wires:
        # Variables named according to spherical coordinate system, it's easier to understand :)    
        radius = inputs[:,w, index['pt']] # corresponding to pt
        azimuth = inputs[:,w, index['phi']] # corresponding to phi
        radius = inputs[:,w, index['pt']] # corresponding to pt
        # Apply rotation gates modulated by the radius (pt) of the particle, which has been scaled to the range [0,1]
        qml.RY(radius*zenith, wires=w)   
        qml.RZ(radius*azimuth, wires=w)  
    
    
    # for phi,theta,omega,i in zip(weights[3*N:4*N],weights[4*N:5*N],weights[5*N:],auto_wires):
    #     qml.Rot(phi,theta,omega,wires=[i]) # perform arbitrary rotation in 3D space instead of RX/RY rotation
    
    for item in two_comb_wires:
        #qml.CNOT(wires=item)
        w=item[1] 
        zenith = inputs[:,w, index['eta']] # corresponding to eta
        azimuth = inputs[:,w, index['phi']] # corresponding to phi
        radius = inputs[:,w, index['pt']] # corresponding to pt
        qml.CRY(radius*zenith,wires=item)
        qml.CRZ(radius*azimuth,wires=item)

    for item in subjet_comb_wires:
        #qml.CNOT(wires=item)
        w=item[0] # Upload the subjet features a third time 
        zenith = inputs[:,w, index['eta']] # corresponding to eta
        azimuth = inputs[:,w, index['phi']] # corresponding to phi
        radius = inputs[:,w, index['pt']] # corresponding to pt
        qml.CRY(radius*zenith,wires=item)
        qml.CRZ(radius*azimuth,wires=item)

    for phi,theta,omega,i in zip(weights[3*N:4*N],weights[4*N:5*N],weights[5*N:],auto_wires):
        qml.Rot(phi,theta,omega,wires=[i]) # perform arbitrary rotation in 3D space instead of RX/RY rotation
    
    # FIXED: handle the case where ancillary_wires is an integer, which is the case when separate_ancilla is False
    if len(ancillary_wires)==1:
        ancillary_wirelist=ancillary_wires*len(ref_wires)
    else:
        ancillary_wirelist=ancillary_wires 
    # SWAP test to measure fidelity
    for ref_wire,trash_wire,ancilla in zip(ref_wires,auto_wires[-n_trash_qubits:],ancillary_wirelist):
        qml.Hadamard(ancilla)
        qml.CSWAP(wires=[ancilla, ref_wire, trash_wire])
        qml.Hadamard(ancilla)
    return qml.expval(qml.operation.Tensor(*[qml.PauliZ(i) for i in ancillary_wires]))
    #import pdb;pdb.set_trace()
    #fidelities = [qml.PauliZ(i) for i in ancillary_wires]
    #return [qml.probs(i) for i in ancillary_wires]
    #return qml.expval(qml.sum(*fidelities)/len(ancillary_wires)) 
class QuantumAutoencoder:
    """
    A class that constructs a Quantum Autoencoder (QAE) using pre-defined circuits.

    Args:
        wires (int): Number of qubits to use.
        shots (int): Number of shots for measurements on quantum states.
        trash_qubits (int): Number of trash qubits used in the circuit.
        dev_name (str): Name of the quantum device to use.
        backend_name (str): Backend for the QNode (e.g., 'autograd', 'torch', 'jax').
        test (bool): If True, sets the circuit immediately for testing.
        separate_ancilla (bool): If True, uses a separate ancillary qubit for each trash/reference qubit pair.
    """
    def __init__(self, wires:int=4,shots=5000,trash_qubits:int=0,dev_name:str='default.qubit',backend_name:str='autograd',test=False,separate_ancilla=False):
        initialize(wires=wires,trash_qubits=trash_qubits,separate_ancilla=separate_ancilla)
        self.device=set_device(shots=shots,device_name=dev_name)
        self.backend=backend_name
        self.current_weights=None
        self.circuit = None
        if test: self.set_circuit() # Set the circuit for inference
    def set_circuit(self,reuploading:bool=False)->None:
        """
        Configures the QNode circuit for the quantum autoencoder.

        Args:
            reuploading (bool): If True, uses the feature re-uploading circuit.
        """
        if reuploading:
            print("Using Feature-Reuploading Circuit with 2 layers")
            self.circuit = qml.QNode(reuploading_circuit,self.device,interface=self.backend)
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
    def run_inference(self,data:np.ndarray,loss_fn:Callable=None):
        """
        Runs inference on the autoencoder circuit using the loaded weights.

        Args:
            data (np.ndarray): Input data for the quantum circuit.
            loss_fn (Callable): Loss function to calculate the quantum cost.

        Returns:
            Tuple[float, float]: Inference loss and fidelity.
        """

        print("Running in inference mode \n No batching will be performed so don't expect a progress bar")
        if self.current_weights is None:
            raise ValueError('Weights not initialized. Load a model first by calling load_weights(model_path)')       
        costs,fids=loss_fn(self.current_weights,inputs=data,quantum_cost=self.circuit,return_fid=True)
        print("Done")
        return costs,fids
    
class QuantumTrainer():
    """
    A class for training a given quantum circuit.

    Args:
        model (QuantumAutoencoder): The quantum autoencoder model to be trained.
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
    def __init__(self, model: QuantumAutoencoder, lr: float = 0.001, optimizer: Callable = None, 
                 loss_fn: Callable = None, save: bool = True, train_max_n: int = 100000, 
                 valid_max_n: int = 20000, epochs: int = 20, patience: int = 4,wandb=None, **kwargs: Any) -> None:
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
        self.optim=optimizer
        self.quantum_loss=loss_fn
        self.current_epoch=0
        self.is_evictable=False
        self.wandb=wandb
        self.history={'train':[],'val':[],'accuracy':[]}
        print (f'Performing optimization with: {self.optim} | Setting Learning rate: {lr}')
        print ('Backend:',self.backend,'\n')

    def iteration(self,data: np.ndarray, train: bool = False) -> Union[float, Tuple[float, float]]:
        """
        Performs a single training or validation iteration.

        Args:
            data (np.ndarray): Batch of input data.
            train (bool): Whether to perform training (True) or validation (False).

        Returns:
            Union[float, Tuple[float, float]]: Training loss (or validation loss and fidelity).
        """
        if train:
            self.current_weights, cost = self.optim.step_and_cost(self.quantum_loss,self.current_weights,inputs=data,quantum_cost=self.circuit)
            return float(cost)
        else: 
            cost,fid=self.quantum_loss(self.current_weights,inputs=data,quantum_cost=self.circuit,return_fid=True)
            return float(cost),float(fid)
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
            for data,label in tqdm(val_loader,total=int(self.valid_max_n/self.batch_size)):
                loss,batch_fid=self.iteration(data,train=False)
                val_loss+=loss
                val_batch_yield+=1
            val_loss=val_loss/val_batch_yield
            if self.wandb is not None:
                self.wandb.log({'val_loss': val_loss})
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
        if name is None:
            if self.current_epoch>100: name = 'ep{:03}.pickle'.format(self.current_epoch)
            else: name='ep{:02}.pickle'.format(self.current_epoch)
        if 'trained' not in name: save_dir=self.checkpoint_dir
        ut.Pickle({'weights':self.current_weights},name,path=save_dir)
    
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

