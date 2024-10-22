import numpy as np
from re import sub
import os,glob,h5py,tqdm,pathlib
import matplotlib;matplotlib.use("Agg")
import matplotlib.pyplot as plt
import processing_functions as pf
from multiprocessing import Pool
from time import sleep
def process_sideband(filename):
    
    deta_jj = 1.4
    
    jPt = 100.
    dt = h5py.special_dtype(vlen=str)
    ipath='/web/abal/public_html/debug/'
    #raw_file_dir = '/ceph/bmaier/CASE/delphes/events/qcd_sqrtshatTeV_13TeV_PU40_NEW_EXT_sideband_parts/'
    outfolder = '/ceph/abal/QML/delphes/substructure/CA_decluster/qcd_sqrtshatTeV_13TeV_PU40_NEW_EXT_sideband_parts/'
    # outfolder = '/ceph/abal/QML/delphes/qcd_sqrtshatTeV_13TeV_PU40_NEW_EXT_sideband_parts/substructure_labels_10/'
    pathlib.Path(outfolder).mkdir(parents=True,exist_ok=True)
    #for filename in tqdm.tqdm(file_paths):
    with h5py.File(filename, "r") as f:

        
        
        localfile = filename.split("/")[-1].replace('.h5','')
        
        outfile = os.path.join(outfolder,localfile+'.h5')
        dEta=f["eventFeatures"][()][:,9]
        
        side_mask = (np.abs(dEta) > deta_jj) 
        
        side_jj=f['eventFeatures'][()]
        side_jj = side_jj[side_mask]

        side_pf=f['jetConstituentsList'][()] # Shape: (N,2,100,3)
        side_pf=side_pf[side_mask]

        #side_pf_flat=np.reshape(side_pf,(-1,100,3)) # for jet-wise training and not event-wise
        side_pf_pxpypzE=np.zeros((side_pf.shape[0],side_pf.shape[1],side_pf.shape[2],4))
        
        side_pf_pxpypzE[...,0]=side_pf[...,2]*np.cos(side_pf[...,1])
        side_pf_pxpypzE[...,1]=side_pf[...,2]*np.sin(side_pf[...,1])
        side_pf_pxpypzE[...,2]=side_pf[...,2]*np.sinh(side_pf[...,0])
        side_pf_pxpypzE[...,3]=np.sqrt(side_pf_pxpypzE[...,0]**2+side_pf_pxpypzE[...,1]**2+side_pf_pxpypzE[...,2]**2)
        
        jet_PFCands,jet_labels, mask, num_PFCands_subleading_jet=pf.uncluster(side_pf_pxpypzE,min_pt=jPt)
        
        # For jets with more than 5 PFCands in the subleading jet, we only consider the 5 leading PFCands
        # If less, then set mask to NUM_SELECTED_PFCANDS - num_PFCands_subleading_jet
        # The jet labels array contains PFCand labels starting from 0 for both leading and subleading jets
        # Something like: [0,1,2,3,4,5,.....,50,0,1,2,......34,999,999,999...] assuming that subjet 1 has 50 PFCands and subjet 2 has 34 PFCands, followed by zero padding to make the shape 100
        # Now applying the mask as jet_labels<mask_limit[:,None] will allow us to select only the leading 5 PFCands of each sub-jet, and if the subleading jet
        # has less than 5 PFCands, then we select the balance PFCands from the leading jet

        #mask_limit=np.where(num_PFCands_subleading_jet>5,5,NUM_SELECTED_PFCANDS-num_PFCands_subleading_jet)
        
        #pf_mask=jet_labels<mask_limit[...,None]
        side_jj=side_jj[mask]
        side_truth=np.zeros_like(side_jj[:,0])
        
        #import pdb;pdb.set_trace()
        side_jj=side_jj[mask]
        side_truth=np.zeros_like(side_jj[:,0])
        side_hf = h5py.File(outfile, 'w')
        side_hf.create_dataset('particleFeatures', data=jet_PFCands)
        side_hf.create_dataset('eventFeatures', data=side_jj)
        side_hf.create_dataset('particleFeatureNames', data=np.array(f['particleFeatureNames'][()],dtype=dt))
        side_hf.create_dataset('eventFeatureNames', data=np.array(f['eventFeatureNames'][()],dtype=dt))
        side_hf.create_dataset('truth_label', data=side_truth)
        side_hf.create_dataset('truth_label_shape', data=side_truth.shape)
        side_hf.create_dataset('num_PFCands_subleading_jet',data=num_PFCands_subleading_jet)
        side_hf.create_dataset('PFCand_subjet_labels',data=jet_labels)
        side_hf.close()


if __name__ == '__main__':
    file_paths=glob.glob('/ceph/bmaier/CASE/delphes/events/qcd_sqrtshatTeV_13TeV_PU40_NEW_EXT_sideband_parts/*.h5')
    num_cores=min(len(file_paths), os.cpu_count()//2)
    print(f"Will use {num_cores} cores to process {len(file_paths)} files")
    pool = Pool(num_cores);sleep(3)
    pool.map(process_sideband, file_paths) 