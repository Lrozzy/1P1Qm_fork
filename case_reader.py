'''

Date: August 2024
Author: Aritra Bal, ETP
Description: This script contains the data loader classes for the Delphes Datasets derived from the CASEDelphes analysis EXO-22-026

'''

import os,h5py,glob
os.environ["PYTHONPATH"]='/work/abal/qae_hep/'
import helpers.utils as ut

from pennylane import numpy as np
from typing import List

from sklearn.preprocessing import MinMaxScaler
from utils import print_events
import __main__
import numpy as nnp
import torch
from torch.utils.data import IterableDataset, DataLoader

class H5IterableDataset(IterableDataset):
    def __init__(self, data_dir:str=ut.path_dict['QCD_train'], data_key:str='eventFeatures',max_samples:int=5e4,epsilon=1.0e-4, num_particles=5):
        """
        Args:
            data_dir (str): Directory containing the .h5 files.
            data_key (str): Key to access the numpy array inside the .h5 files.

        Yields:
            torch.Tensor: A batch of data samples.
        Notes:
            - Ignore this class, superseded by CASEDelphesJetDataset
        """
        self.data_dir = data_dir
        self.data_key = data_key
        self.file_paths = sorted(glob.glob(os.path.join(data_dir, '*.h5')))
        self.max_samples = max_samples
        self.epsilon=epsilon
        self.num_particles=num_particles
        self.batch_counter=0
    def __iter__(self):
        sample_counter=0
        for file_path in self.file_paths:
            with h5py.File(file_path, 'r') as file:
                data = file[self.data_key][:,:self.num_particles,:] # Read up to num_particles particles per event, h5 file contains 25 particles per event
                pt_index=ut.getIndex('particle','pt')
                scaler = MinMaxScaler(feature_range=(0, np.pi-self.epsilon))
                data_shape = data.shape
                data_reshaped = data[:, :, pt_index].flatten()[:, np.newaxis]
                data_scaled = scaler.fit_transform(data_reshaped)
                data[:, :, -1] = data_scaled.reshape(data_shape[0], data_shape[1])
                
                # Check if adding this file's samples would exceed the max_samples limit
                if self.max_samples is not None and sample_counter + data.shape[0] > self.max_samples:
                    limit = self.max_samples - sample_counter
                    sample_counter=self.max_samples
                    yield torch.from_numpy(data[:limit])
                    return
                else:
                    yield torch.from_numpy(data)
                    sample_counter += data.shape[0]

                # Stop yielding if we've reached the max_samples
                if self.max_samples is not None and sample_counter >= self.max_samples:
                    return # don't use break here

class CASEDelphesJetDataset(IterableDataset):
    def __init__(self, filelist:list[str]=None, batch_size:int=32, max_samples:int=5e4, data_key='jetConstituentsList',\
                 feature_key='eventFeatures',input_shape:tuple[int]=(10, 3),epsilon:float=1.0e-4,train:bool=True,yield_energies=False):
        super().__init__()
        self.filelist = sorted(filelist)
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.pt_index=ut.getIndex('particle','pt')
        self.eta_index=ut.getIndex('particle','eta')
        self.phi_index=ut.getIndex('particle','phi')
        self.j1E_index=ut.getIndex('event','j1E')
        self.j2E_index=ut.getIndex('event','j2E')
        self.mjj_index=ut.getIndex('event','mJJ')
        self.max_samples = 2*max_samples
        self.epsilon=epsilon
        self.data_key=data_key
        self.feature_key=feature_key
        self.train=train
        self.batch_counter=0
        self.yield_energies=yield_energies
    def rescale(self, data, min=0., max=1., epsilon=1.0e-4):
        data_shape = data.shape
        max-=epsilon
        #print(f"pt_index: {self.pt_index}, eta_index: {self.eta_index}, phi_index: {self.phi_index}")
        scaler = MinMaxScaler(feature_range=(min, max))
        data_reshaped = data.flatten()[:, np.newaxis]
        data_scaled = scaler.fit_transform(data_reshaped)
        return data_scaled.reshape(data_shape[0], data_shape[1])
    
    
    def load_and_preprocess_file(self, file_path,inference=False):
        '''
        Args: 
            file_path (str): Path to the .h5 file containing the data.
        Returns:
            np.ndarray: Stacked data from both jets.
            np.ndarray: Stacked truth labels.
        '''
        
        with h5py.File(file_path, 'r') as file:
            # Read data of shape (N,2,100,3) where N is the number of events, 2 is the number of jets, 100 is the number of particles and 3 is the (eta,phi,pt) of each particle
            jet_etaphipt = np.array(file[self.data_key][:,:, :self.input_shape[0], :]) # because input_shape[0] is the number of particles
            #jet1_energy=np.array(file[self.feature_key][:,self.j1E_index])
            #jet2_energy=np.array(file[self.feature_key][:,self.j2E_index])
            mjj=np.array(file[self.feature_key][:,self.mjj_index])
            
            try:
                truth_label = np.array(file['truth_label'][()])
            except:
                #print("Truth labels not found in file. Inferring")
                if 'qcd'.casefold() in file_path.casefold(): # case-insensitive search
                    #print("Inferred --> QCD. Setting label to 0")
                    truth_label = np.zeros(jet_etaphipt.shape[0])
                else:
                    #print("Inferred --> non-QCD. Setting label to 1")
                    truth_label = np.ones(jet_etaphipt.shape[0])
        if inference:
            return jet_etaphipt,mjj,truth_label
        
        stacked_data = np.reshape(jet_etaphipt,newshape=[-1, jet_etaphipt.shape[2], jet_etaphipt.shape[3]])
        # NOTE: arg newshape is deprecated in numpy>=2.10.0
        stacked_labels = np.concatenate([truth_label, truth_label], axis=0)
        #stacked_energies = np.concatenate([jet1_energy, jet2_energy], axis=0)
        stacked_data[:, :, self.pt_index] = self.rescale(stacked_data[:, :, self.pt_index], min=0., max=1.0, epsilon=self.epsilon)
        stacked_data[:, :, self.eta_index] = self.rescale(stacked_data[:, :, self.eta_index], min=0., max=np.pi, epsilon=self.epsilon)
        stacked_data[:, :, self.phi_index] = self.rescale(stacked_data[:, :, self.phi_index], min=-np.pi, max=np.pi, epsilon=self.epsilon)
        
        return stacked_data, stacked_labels#,stacked_energies
    
    def load_for_inference(self):
        self.max_samples = self.max_samples//2
        mjj_array=[]
        while len(mjj_array)<self.max_samples:
            for i,file_path in enumerate(self.filelist):
                jet_etaphipt,mjj,truth_label= self.load_and_preprocess_file(file_path,inference=True)
                if i==0:
                    mjj_array=mjj
                    jet_array=jet_etaphipt
                    truth_labels=truth_label
                else:
                    mjj_array=np.concatenate([mjj_array,mjj],axis=0)
                    jet_array=np.concatenate([jet_array,jet_etaphipt],axis=0)
                    truth_labels=np.concatenate([truth_labels,truth_label],axis=0)
                if len(mjj_array)>=self.max_samples:
                    break
        jet_array=jet_array[:self.max_samples]
        mjj_array=mjj_array[:self.max_samples]
        truth_labels=truth_labels[:self.max_samples]
        return jet_array[:,0,:,:],jet_array[:,1,:,:],mjj_array,truth_labels

    def __iter__(self):
        sample_counter=0
        #self.batch_counter=0
        for file_path in self.filelist:
            data, labels = self.load_and_preprocess_file(file_path)
            # Shuffle the data from this file only if training on a per-jet basis, otherwise it becomes necessary to preserve the order of jets
            
            indices = np.arange(data.shape[0])
            np.random.shuffle(indices)
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
                #batch_jet_energy = jet_energy[i:end]
                
                # Convert data to PyTorch tensors before yielding
                batch_data = torch.from_numpy(batch_data).float()
                batch_labels = torch.from_numpy(batch_labels).float() # Assuming labels are floats
                #batch_jet_energy = torch.from_numpy(batch_jet_energy).float()
                
                sample_counter += batch_data.shape[0]
                #self.batch_counter+=1
                #if self.train and self.yield_energies:
                #    yield np.array(batch_data,requires_grad=False), np.array(batch_jet_energy,requires_grad=False)
                if self.train:
                    yield np.array(batch_data,requires_grad=False)
                else:
                    yield np.array(batch_data,requires_grad=False), np.array(batch_labels,requires_grad=False)

# def collate_fn(samples):
#     if type(samples[0])==tuple:
#         X,y=[],[]
#         for item in samples: X.append(item[0]),y.append(item[1])
#         X,y=np.array(X),np.array(y).flatten()
#         return X,y
#     else: 
        # return np.array(samples,requires_grad=False)
    
def CASEDelphesDataLoader(filelist:list[str]=None,batch_size:int=128, input_shape:tuple[int]=(100, 3),train:bool=True,max_samples:int=5e4):
    '''
    Args:
        filelist (list[str]): List of file paths to the .h5 files containing the data.
        batch_size (int): Number of samples in each batch.
        shuffle_buffer_size (int): Number of samples to shuffle in the buffer.
        input_shape (tuple[int]): Shape of the input data, with the first element being the no. of particles per jet to read and the 2nd typically being (eta,phi,pt).
    Returns:
        DataLoader: Torch DataLoader object that yields batches of data.
    '''
    print(f"Will read only first {input_shape[0]} particles per jet")
    dataset = CASEDelphesJetDataset(filelist=filelist, batch_size=batch_size, input_shape=input_shape,train=train,max_samples=max_samples)
    
    return DataLoader(dataset, batch_size=None)  # None for batch_size since batching is managed by the dataset
def rescale_and_reshape(data: list[np.ndarray]):
    '''
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
    stacked_data[:, :, eta_index] = rescale(stacked_data[:, :, eta_index], min=0., max=nnp.pi, epsilon=1.0e-4)
    stacked_data[:, :, phi_index] = rescale(stacked_data[:, :, phi_index], min=-nnp.pi, max=nnp.pi, epsilon=1.0e-4)
    #import pdb;pdb.set_trace()
    import pdb;pdb.set_trace()
    return nnp.split(stacked_data, data_len)[:-1]

def rescale(data, min=0., max=1., epsilon=1.0e-4):
    data_shape = data.shape
    max-=epsilon
    #print(f"pt_index: {self.pt_index}, eta_index: {self.eta_index}, phi_index: {self.phi_index}")
    scaler = MinMaxScaler(feature_range=(min, max))
    data_reshaped = data.flatten()[:, np.newaxis]
    data_scaled = scaler.fit_transform(data_reshaped)
    return data_scaled.reshape(data_shape[0], data_shape[1])