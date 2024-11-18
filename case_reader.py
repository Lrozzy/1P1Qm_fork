'''
Date: August 2024
Author: Aritra Bal, ETP
Description: This script contains the data loader classes for the MC Datasets used in the studies performed for the CMS analysis EXO-22-026
'''

import h5py
import helpers.utils as ut
from pennylane import numpy as np
from sklearn.preprocessing import MinMaxScaler
import __main__
import numpy as nnp
import torch
from torch.utils.data import IterableDataset, DataLoader
from typing import List, Tuple, Union
import loguru
eta_lims=ut.feature_limits['eta']
phi_lims=ut.feature_limits['phi']
class CASEDelphesJetDataset(IterableDataset):
    """
    Iterable dataset class to load jet data from .h5 files.

    Args:
        filelist (List[str]): List of file paths to the .h5 files.
        batch_size (int): Number of samples in each batch.
        max_samples (int): Maximum number of samples to load.
        data_key (str): Key to access the jet data inside the .h5 files.
        feature_key (str): Key to access event-level features.
        input_shape (tuple[int]): Input shape specifying number of particles and features.
        epsilon (float): Small constant to avoid division by zero.
        train (bool): Whether to yield only training data, or labels as well.
        yield_energies (bool): If True, yield jet energies as well.

    Yields:
        np.ndarray: Batch of data samples.
    """
    def __init__(self, filelist:List[str]=None, batch_size:int=32, max_samples:int=5e4, data_key='jetConstituentsList',\
                 feature_key='eventFeatures',input_shape:tuple[int]=(10, 3),epsilon:float=1.0e-4,train:bool=True,yield_energies=False,\
                    normalize_pt:bool=False,use_subjet_PFCands:bool=False,logger=None):
        super().__init__()
        self.filelist = sorted(filelist)
        self.dataset='delphes'
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.pt_index=ut.getIndex('particle','pt')
        self.eta_index=ut.getIndex('particle','eta')
        self.phi_index=ut.getIndex('particle','phi')
        
        self.use_subjet_PFCands=use_subjet_PFCands
        self.max_samples = 2*max_samples
        self.epsilon=epsilon
        self.data_key=data_key
        self.feature_key=feature_key
        self.train=train
        self.batch_counter=0
        self.normalize_pt=normalize_pt
        self.logger=logger
        self.n_qubits=input_shape[0]
    def set_dataset_type(self,dataset:str):
        self.dataset=dataset
        if self.dataset=='delphes':
            self.j1pt_index=ut.getIndex('event','j1Pt')
            self.j2pt_index=ut.getIndex('event','j2Pt') # Bug fix
            self.mjj_index=ut.getIndex('event','mJJ')
        elif self.dataset=='jetclass':
            self.jpt_index=ut.getIndex('jet','jet_pt')
            self.j_msd_index=ut.getIndex('jet','jet_sdmass')
        else:
            raise ValueError("Dataset type not recognized. Must be either delphes or jetclass")
    def fixed_rescale(self,data: np.ndarray, epsilon: float = 1.0e-4, type='pt') -> np.ndarray:
        """
        Rescales the data to a specified range. Instead of using the min/max values of the data array,
        uses fixed values instead, which can be modified in the assumed_limits array

        Args:
            data (np.ndarray): Input data to rescale.
            epsilon (float): Small offset to prevent numerical instability.

        Returns:
            np.ndarray: Rescaled data.
        """
        min=ut.feature_limits[type]['min']
        max=ut.feature_limits[type]['max']
        max-=epsilon
        assumed_limits=ut.assumed_limits
        if type not in ['pt','eta','phi']:
            raise NameError("Type must be either of [pt,eta,phi]")
        if self.logger is not None:
            self.logger.info(f"Assuming fixed sample maxima: [{assumed_limits[type][0]},{assumed_limits[type][1]}] for variable {type}")
        else:
            print(f"Assuming fixed sample maxima: [{assumed_limits[type][0]},{assumed_limits[type][1]}] for variable {type}")
        data_shape = data.shape
        data_reshaped = data.flatten()
        data_scaled = ((data_reshaped - assumed_limits[type][0])/(assumed_limits[type][1]-assumed_limits[type][0]))*(max-min) + min # scale using fixed values of 
        return data_scaled.reshape(data_shape[0], data_shape[1])
    
    def load_and_preprocess_file(self, file_path:str,inference:bool=False):
        """
        Loads and preprocesses a single .h5 file.

        Args:
            file_path (str): Path to the .h5 file.
            inference (bool): Whether the data is being loaded for inference.

        Returns:
            np.ndarray: Stacked data from both jets.
            np.ndarray: Stacked truth labels.
        """
        
        with h5py.File(file_path, 'r') as file:
            # Read data of shape (N,2,100,3) where N is the number of events, 2 is the number of jets, 100 is the number of particles and 3 is the (eta,phi,pt) of each particle
            mjj=np.array(file[self.feature_key][:,self.mjj_index])
            j1pt=np.array(file[self.feature_key][:,self.j1pt_index])
            j2pt=np.array(file[self.feature_key][:,self.j2pt_index])

            if self.use_subjet_PFCands:
                if self.logger is not None:
                    self.logger.info(f"Reading {self.n_qubits//2} PFCands from each subjet for a total of {self.n_qubits} PFCands per jet")
                else:
                    print(f"Reading {self.n_qubits//2} PFCands from each subjet for a total of {self.n_qubits} PFCands per jet")
                jet_etaphipt = np.array(file[self.data_key][()]) # because n_qubits is the number of particles
                num_PFCands_subleading_jet=file['num_PFCands_subleading_jet'][()]
                evt_subjet_idx=file['PFCand_subjet_idx'][()] # N x 2 x 100
                mask_limit=np.where(num_PFCands_subleading_jet>self.n_qubits//2,self.n_qubits//2,self.n_qubits-num_PFCands_subleading_jet) # N x 2
                pf_mask=evt_subjet_idx<mask_limit[...,None] # N x 2 x 100
                jet_etaphipt=jet_etaphipt[pf_mask].reshape(-1,2,self.n_qubits,3) # N x 2 x n_qubits x 3
            else:
                if self.logger is not None:
                    self.logger.info(f"Reading hardest {self.n_qubits} PFCands per jet")
                else:
                    print(f"Reading hardest {self.n_qubits} PFCands per jet")
                jet_etaphipt = np.array(file[self.data_key][()]) # because n_qubits is the number of particles
                sorted_indices = np.argsort(-jet_etaphipt[...,2], axis=2)
                jet_etaphipt = np.take_along_axis(jet_etaphipt, sorted_indices[...,None], axis=2)
                jet_etaphipt = jet_etaphipt[:,:,:self.n_qubits,:]
            #jet1_energy=np.array(file[self.feature_key][:,self.j1E_index])
            #jet2_energy=np.array(file[self.feature_key][:,self.j2E_index])
            
            try:
                truth_label = np.array(file['truth_label'][()])
            except:
                if 'qcd'.casefold() in file_path.casefold(): # case-insensitive search
                    truth_label = np.zeros(jet_etaphipt.shape[0])
                else:
                    truth_label = np.ones(jet_etaphipt.shape[0])
        if self.normalize_pt:
            print("Normalizing PFCand pT by jet pT")
            jet_etaphipt[:,0,:,self.pt_index]=jet_etaphipt[:,0,:,self.pt_index]/j1pt[:,np.newaxis]
            jet_etaphipt[:,1,:,self.pt_index]=jet_etaphipt[:,1,:,self.pt_index]/j2pt[:,np.newaxis]
        else:
            jet_etaphipt[:,0,:,self.pt_index]=self.fixed_rescale(jet_etaphipt[:,0,:,self.pt_index], epsilon=self.epsilon,type='pt')
            jet_etaphipt[:,1,:,self.pt_index]=self.fixed_rescale(jet_etaphipt[:,1,:,self.pt_index], epsilon=self.epsilon,type='pt')
        
        jet_etaphipt[:,0,:,self.eta_index]=self.fixed_rescale(jet_etaphipt[:,0,:,self.eta_index], epsilon=self.epsilon,type='eta')
        jet_etaphipt[:,0,:,self.phi_index]=self.fixed_rescale(jet_etaphipt[:,0,:,self.phi_index], epsilon=self.epsilon,type='phi')
        jet_etaphipt[:,1,:,self.eta_index]=self.fixed_rescale(jet_etaphipt[:,1,:,self.eta_index], epsilon=self.epsilon,type='eta')
        jet_etaphipt[:,1,:,self.phi_index]=self.fixed_rescale(jet_etaphipt[:,1,:,self.phi_index], epsilon=self.epsilon,type='phi')
            
        # If you only wish to test, then no need to batch, just return the entire array
        if inference:
            return jet_etaphipt,np.stack([mjj,j1pt,j2pt],axis=-1),truth_label
        
        stacked_data = np.reshape(jet_etaphipt,newshape=[-1, jet_etaphipt.shape[2], jet_etaphipt.shape[3]])
        stacked_labels = np.concatenate([truth_label, truth_label], axis=0)
        
        print("sample max pt: ",np.max(stacked_data[:,:,self.pt_index]))
        print("sample min pt: ",np.min(stacked_data[:,:,self.pt_index]))
        print("sample max eta: ",np.max(stacked_data[:,:,self.eta_index]))
        print("sample min eta: ",np.min(stacked_data[:,:,self.eta_index]))
        print("sample max phi: ",np.max(stacked_data[:,:,self.phi_index]))
        print("sample min phi: ",np.min(stacked_data[:,:,self.phi_index]))
        return stacked_data, stacked_labels#,stacked_energies
    
    def load_for_inference(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
    Loads data from the specified .h5 files for inference.

    This method processes the jet and event data from the files, ensuring that 
    only a maximum number of samples specified by `max_samples` are loaded. 
    It returns separate arrays for the jet features of the two jets in each event,
    the dijet invariant mass (mjj), and the corresponding truth labels.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
            - jet_array[:,0,:,:] (np.ndarray): Jet features for the first jet in each event, with shape (N, num_particles, 3).
            - jet_array[:,1,:,:] (np.ndarray): Jet features for the second jet in each event, with shape (N, num_particles, 3).
            - jetFeatures_array (np.ndarray): (mjj,j1pt,j2pt) values for each event, with shape (N,3).
            - truth_labels (np.ndarray): Ground truth labels for each event, with shape (N,).

    Notes:
        - This method halves the number of samples from `max_samples` to avoid double counting, because we wish to return 2 jets per event
    """
        self.max_samples = self.max_samples//2
        print(f"Will read a total of {self.max_samples} events for inference")
        jetFeatures_array=[]
        while len(jetFeatures_array)<self.max_samples:
            for i,file_path in enumerate(self.filelist):
                jet_etaphipt,jet_features,truth_label= self.load_and_preprocess_file(file_path,inference=True)
                if i==0:
                    jetFeatures_array=jet_features
                    jet_array=jet_etaphipt
                    truth_labels=truth_label
                else:
                    jetFeatures_array=np.concatenate([jetFeatures_array,jet_features],axis=0)
                    jet_array=np.concatenate([jet_array,jet_etaphipt],axis=0)
                    truth_labels=np.concatenate([truth_labels,truth_label],axis=0)
                if len(jetFeatures_array)>=self.max_samples:
                    break
        jet_array=jet_array[:self.max_samples]
        jetFeatures_array=jetFeatures_array[:self.max_samples]
        truth_labels=truth_labels[:self.max_samples]
        return jet_array[:,0,:,:],jet_array[:,1,:,:],jetFeatures_array,truth_labels

    def __iter__(self)-> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        '''
        Iterator that yields batches of data
        '''
        sample_counter=0
        #self.batch_counter=0
        for file_path in self.filelist:
            data, labels = self.load_and_preprocess_file(file_path)
            # Shuffle the data from this file only if training on a per-jet basis, otherwise it becomes necessary to preserve the order of jets
            
            indices = np.arange(data.shape[0])
            np.random.shuffle(indices); print("Loaded data was shuffled")
            data = data[indices]
            labels = labels[indices]
            
            # Yield data in batches
            for i in range(0, len(data), self.batch_size):
                if sample_counter >= self.max_samples:
                    return
                
                end = i + self.batch_size
                if end > len(data):
                    end = len(data)
                
                batch_data = data[i:end]
                batch_labels = labels[i:end]
                
                batch_data = torch.from_numpy(batch_data).float()
                batch_labels = torch.from_numpy(batch_labels).float() # Assuming labels are floats
                
                sample_counter += batch_data.shape[0]
                if self.train:
                    yield np.array(batch_data,requires_grad=False)
                else:
                    yield np.array(batch_data,requires_grad=False), np.array(batch_labels,requires_grad=False)

def CASEDelphesDataLoader(input_shape:tuple[int]=(100, 3),train:bool=True,use_subjet_PFCands:bool=False,dataset='delphes',**kwargs) -> DataLoader:
    '''
    Wrapper function to create a DataLoader for the CASEDelphesJetDataset.
    Args:
        filelist (List[str]): List of file paths to the .h5 files containing the data.
        batch_size (int): Number of samples in each batch.
        shuffle_buffer_size (int): Number of samples to shuffle in the buffer.
        input_shape (tuple[int]): Shape of the input data, with the first element being the no. of particles per jet to read and the 2nd typically being (eta,phi,pt).
    Returns:
        DataLoader: Torch DataLoader object that yields batches of data.
    '''
    print(f"Will read only {input_shape[0]} particles per jet")
    if use_subjet_PFCands:
        print("Equally divided between 2 subjets per jet")
    
    dset = CASEDelphesJetDataset(input_shape=input_shape,train=train,use_subjet_PFCands=use_subjet_PFCands,**kwargs)
    dset.set_dataset_type(dataset)
    return DataLoader(dset, batch_size=None)  # None for batch_size since batching is managed by the dataset












########## REDUNDANT STUFF ############



def fixed_rescale(data: np.ndarray, min: float = 0.0, max: float = 1.0, epsilon: float = 1.0e-4, type='pt') -> np.ndarray:
    """
    Rescales the data to a specified range. 
    Does not use sample min/max values, but fixed values instead.

    Args:
        data (np.ndarray): Input data to rescale.
        min (float): Minimum value of the scaled data.
        max (float): Maximum value of the scaled data.
        epsilon (float): Small offset to prevent numerical instability.

    Returns:
        np.ndarray: Rescaled data.
    """
    assumed_limits={'pt':[epsilon,3000.],'eta':[-0.8,0.8],'phi':[-0.8,0.8]}
    if type not in ['pt','eta','phi']:
        raise NameError("Type must be either of [pt,eta,phi]")
    
    data_shape = data.shape
    data_scaled = ((data - assumed_limits[type][0])/(assumed_limits[type][1]-assumed_limits[type][0]))*(max-min) + min # scale using fixed values of 
    #import pdb;pdb.set_trace()
    
    return data_scaled.reshape(data_shape[0], data_shape[1])

def fixed_rescale_and_reshape(data: np.ndarray)-> np.ndarray:
    '''
    Rescales the input data for a given jet PFCand array.

    Args: 
        data (np.ndarray): Data of shape [N,n_inputs,3] to be rescaled
    Returns:
        np.ndarray: Data rescaled for the input jet array
        Note that the rescaling is performed with a fixed max/min value for each variable
    '''
    pt_index=ut.getIndex('particle','pt')
    eta_index=ut.getIndex('particle','eta')
    phi_index=ut.getIndex('particle','phi')
    
    data[:, :, pt_index] = fixed_rescale(data[:, :, pt_index], min=0., max=1.0, epsilon=1.0e-4)
    data[:, :, eta_index] = fixed_rescale(data[:, :, eta_index], min=0., max=nnp.pi, epsilon=0,type='eta')
    data[:, :, phi_index] = fixed_rescale(data[:, :, phi_index], min=-nnp.pi, max=nnp.pi, epsilon=0,type='phi')
    return data

def rescale_and_reshape(data: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Rescales and reshapes the input data for jets 1 and 2.

    Args: 
        data (np.ndarray): Data of shape [N,2,n_inputs,3] to be rescaled and reshaped.
    Returns:
        np.ndarray,np.ndarray: Data rescaled and reshaped for jets 1 and 2
        Note that the rescaling is performed first, taking both jets of an event into account and only then is it reshaped into 2 separate jets
    '''
    pt_index=ut.getIndex('particle','pt')
    eta_index=ut.getIndex('particle','eta')
    phi_index=ut.getIndex('particle','phi')
    data_len=nnp.cumsum([arr.shape[0] for arr in data])
    print(data_len)
    
    stacked_data = nnp.concatenate(data, axis=0)
    stacked_data[:, :, pt_index] = rescale(stacked_data[:, :, pt_index], min=0., max=1.0, epsilon=1.0e-4)
    stacked_data[:, :, eta_index] = rescale(stacked_data[:, :, eta_index], min=-nnp.pi, max=nnp.pi, epsilon=0)
    stacked_data[:, :, phi_index] = rescale(stacked_data[:, :, phi_index], min=-nnp.pi, max=nnp.pi, epsilon=0)
    return nnp.split(stacked_data, data_len)[:-1]

def rescale(data: np.ndarray, min: float = 0.0, max: float = 1.0, epsilon: float = 1.0e-4) -> np.ndarray:
    """
    Rescales the data to a specified range.

    Args:
        data (np.ndarray): Input data to rescale.
        min (float): Minimum value of the scaled data.
        max (float): Maximum value of the scaled data.
        epsilon (float): Small offset to prevent numerical instability.

    Returns:
        np.ndarray: Rescaled data.
    """
    data_shape = data.shape
    max-=epsilon
    scaler = MinMaxScaler(feature_range=(min, max))
    data_reshaped = data.flatten()[:, np.newaxis]
    data_scaled = scaler.fit_transform(data_reshaped)
    return data_scaled.reshape(data_shape[0], data_shape[1])
