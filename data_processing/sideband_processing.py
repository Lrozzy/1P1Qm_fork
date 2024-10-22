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
    outfolder = '/ceph/abal/QML/delphes/substructure/CA_decluster/qcd_sqrtshatTeV_13TeV_PU40_NEW_EXT_sideband_parts/'
    pathlib.Path(outfolder).mkdir(parents=True,exist_ok=True)
    
    with h5py.File(filename, "r") as f:
        pf_names=np.array(f['particleFeatureNames'][()],dtype=dt)
        ef_names=np.array(f['eventFeatureNames'][()],dtype=dt)
        dEta=f["eventFeatures"][()][:,9]
        side_jj=f['eventFeatures'][()]
        side_pf=f['jetConstituentsList'][()] # Shape: (N,2,100,3)
    
    sf_names=np.array(['sj1Pt','sj1Eta','sj1Phi','sj1E','sj2Pt','sj2Eta','sj2Phi','sj2E'],dtype=dt)
    
    localfile = filename.split("/")[-1].replace('.h5','')
    
    outfile = os.path.join(outfolder,localfile+'.h5')
    
    
    side_mask = (np.abs(dEta) > deta_jj) 
    
    
    side_jj = side_jj[side_mask]

    
    side_pf=side_pf[side_mask]

    #side_pf_flat=np.reshape(side_pf,(-1,100,3)) # for jet-wise training and not event-wise
    side_pf_pxpypzE=np.zeros((side_pf.shape[0],side_pf.shape[1],side_pf.shape[2],4))
    
    side_pf_pxpypzE[...,0]=side_pf[...,2]*np.cos(side_pf[...,1])
    side_pf_pxpypzE[...,1]=side_pf[...,2]*np.sin(side_pf[...,1])
    side_pf_pxpypzE[...,2]=side_pf[...,2]*np.sinh(side_pf[...,0])
    side_pf_pxpypzE[...,3]=np.sqrt(side_pf_pxpypzE[...,0]**2+side_pf_pxpypzE[...,1]**2+side_pf_pxpypzE[...,2]**2)
    
    jet_PFCands,evt_subjet_idx, mask, num_PFCands_subleading_jet,evt_subjet_features,evt_subjet_labels=pf.uncluster(side_pf_pxpypzE,min_pt=jPt,test=False)
    
    # For jets with more than NUM_SELECTED_PFCANDS//2 PFCands in the subleading jet, we only consider the NUM_SELECTED_PFCANDS//2 leading PFCands
    # If less, then set mask to NUM_SELECTED_PFCANDS - num_PFCands_subleading_jet
    # The jet labels array contains PFCand labels starting from 0 for both leading and subleading jets
    # Something like: [0,1,2,3,4,5,.....,50,0,1,2,......34,999,999,999...] assuming that subjet 1 has 50 PFCands and subjet 2 has 34 PFCands, followed by zero padding to make the shape 100
    # Now applying the mask as evt_subjet_idx<mask_limit[:,None] will allow us to select only the leading NUM_SELECTED_PFCANDS//2 PFCands of each sub-jet, and if the subleading jet
    # has less than NUM_SELECTED_PFCANDS//2 PFCands, then we select the balance PFCands from the leading jet

    #mask_limit=np.where(num_PFCands_subleading_jet>NUM_SELECTED_PFCANDS//2,NUM_SELECTED_PFCANDS//2,NUM_SELECTED_PFCANDS-num_PFCands_subleading_jet)
    
    #pf_mask=evt_subjet_idx<mask_limit[...,None]
    side_jj=side_jj[mask]
    side_truth=np.zeros_like(side_jj[:,0])
    

    with h5py.File(outfile, 'w') as side_hf:
        side_hf.create_dataset('particleFeatures', data=jet_PFCands)
        side_hf.create_dataset('eventFeatures', data=side_jj)
        side_hf.create_dataset('particleFeatureNames', data=pf_names)
        side_hf.create_dataset('eventFeatureNames', data=ef_names)
        side_hf.create_dataset('truth_label', data=side_truth)
        side_hf.create_dataset('truth_label_shape', data=side_truth.shape)
        side_hf.create_dataset('num_PFCands_subleading_jet',data=num_PFCands_subleading_jet)
        side_hf.create_dataset('PFCand_subjet_idx',data=evt_subjet_idx)
        side_hf.create_dataset('subjet_features',data=evt_subjet_features)
        side_hf.create_dataset('subjet_feature_names',data=sf_names)
        side_hf.create_dataset('subjet_labels',data=evt_subjet_labels)
    print("\n\n#### DONE #####\n\n")

if __name__ == '__main__':
    multi=True
    file_paths=glob.glob('/ceph/bmaier/CASE/delphes/events/qcd_sqrtshatTeV_13TeV_PU40_NEW_EXT_sideband_parts/*.h5')
    if multi:
        num_cores=min(len(file_paths), 12)
        print(f"Will use {num_cores} cores to process {len(file_paths)} files")
        pool = Pool(num_cores);sleep(3)
        pool.map(process_sideband, file_paths) 
    else:
        for filename in file_paths:
            process_sideband(filename)