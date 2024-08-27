'''

Date: August 2024
Author: Aritra Bal, ETP
Description: This script contains the data loader classes for the Delphes Datasets derived from the CASE analysis EXO-22-026

'''

import os,h5py,glob
os.environ["PYTHONPATH"]='/work/abal/qae_hep/'
import helpers.utils as ut

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from utils import print_events
import __main__

import torch
from torch.utils.data import IterableDataset, DataLoader

class H5IterableDataset(IterableDataset):
    def __init__(self, data_dir:str=ut.path_dict['QCD_train'], data_key:str='eventFeatures',max_samples:int=1e6,epsilon=1.0e-4, num_particles=5):
        """
        Args:
            data_dir (str): Directory containing the .h5 files.
            data_key (str): Key to access the numpy array inside the .h5 files.

        Yields:
            torch.Tensor: A batch of data samples.
        """
        self.data_dir = data_dir
        self.data_key = data_key
        self.file_paths = sorted(glob.glob(os.path.join(data_dir, '*.h5')))
        self.max_samples = max_samples
        self.epsilon=epsilon
        self.num_particles=num_particles
    def __iter__(self):
        sample_counter=0
        
        for file_path in self.file_paths:
            with h5py.File(file_path, 'r') as file:
                 # Optionally, apply scaling to the last feature across all samples
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
                    break

class CASEJetDataset(IterableDataset):
    def __init__(self, filelist:list[str]=None, batch_size:int=32, data_key='jetConstituentsList',input_shape:tuple[int]=(10, 3),epsilon:float=1.0e-4):
        self.filelist = filelist
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.pt_index=ut.getIndex('particle','pt')
        self.epsilon=epsilon
        self.data_key=data_key
    def load_and_preprocess_file(self, file_path):
        '''
        Args: 
            file_path (str): Path to the .h5 file containing the data.
        Returns:
            np.ndarray: Stacked data from both jets.
            np.ndarray: Stacked truth labels.
        '''
        with h5py.File(file_path, 'r') as file:
            # Read data of shape (N,2,100,3) where N is the number of events, 2 is the number of jets, 100 is the number of particles and 3 is the (eta,phi,pt) of each particle
            jet_pxpypz = np.array(file[self.data_key][:,:, :self.input_shape[0], :]) # because input_shape[0] is the number of particles
            try:
                truth_label = np.array(file['truth_label'][()])
            except:
                print("Truth labels not found in file. Inferring")
                if 'qcd'.casefold() in file_path.casefold(): # Case-insensitive search
                    print("Inferred --> QCD")
                    truth_label = np.zeros(jet_pxpypz.shape[0])
                else:
                    print("Inferred --> non-QCD")
                    truth_label = np.ones(jet_pxpypz.shape[0])
        
        stacked_data = np.reshape(jet_pxpypz,newshape=[-1, jet_pxpypz.shape[2], jet_pxpypz.shape[3]])
        # NOTE: arg newshape is deprecated in numpy>=2.10.0
        stacked_labels = np.concatenate([truth_label, truth_label], axis=0)
        
        scaler = MinMaxScaler(feature_range=(0, np.pi-self.epsilon))
        data_shape = stacked_data.shape
        data_reshaped = stacked_data[:, :, self.pt_index].flatten()[:, np.newaxis]
        data_scaled = scaler.fit_transform(data_reshaped)
        stacked_data[:, :, -1] = data_scaled.reshape(data_shape[0], data_shape[1])
        
        return stacked_data, stacked_labels

    def __iter__(self):
        for file_path in self.filelist:
            data, labels = self.load_and_preprocess_file(file_path)
            
            # Shuffle the data from this file
            indices = np.arange(data.shape[0])
            np.random.shuffle(indices)
            data = data[indices]
            labels = labels[indices]
            
            # Yield data in batches
            for i in range(0, len(data), self.batch_size):
                end = i + self.batch_size
                if end > len(data):
                    end = len(data)
                batch_data = data[i:end]
                batch_labels = labels[i:end]
                
                # Convert data to PyTorch tensors before yielding
                batch_data = torch.from_numpy(batch_data).float()
                batch_labels = torch.from_numpy(batch_labels).float() # Assuming labels are floats
                
                yield batch_data, batch_labels

### Step 2: Create the DataLoader
def CASEDataLoader(filelist:list[str]=None,batch_size:int=128, input_shape:tuple[int]=(100, 3)):
    '''
    Args:
        filelist (list[str]): List of file paths to the .h5 files containing the data.
        batch_size (int): Number of samples in each batch.
        shuffle_buffer_size (int): Number of samples to shuffle in the buffer.
        input_shape (tuple[int]): Shape of the input data, with the first element being the no. of particles per jet to read and the 2nd typically being (eta,phi,pt).
    Returns:
        DataLoader: Torch DataLoader object that yields batches of data.
    '''
    dataset = CASEJetDataset(filelist=filelist, batch_size=batch_size, input_shape=input_shape)
    return DataLoader(dataset, batch_size=None)  # None for batch_size since batching is managed by the dataset
