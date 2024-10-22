import numpy as np
from re import sub
import os,glob,h5py,pathlib,tqdm,re
import matplotlib;matplotlib.use("Agg")
import matplotlib.pyplot as plt
import processing_functions as pf
from multiprocessing import Pool
from time import sleep
signal='AtoHZ'#'qcd_sqrtshatTeV_13TeV_PU40_NEW_EXT_signalregion_parts'#
def process_signal_region(filename):
    dt = h5py.special_dtype(vlen=str)
    ipath='/web/abal/public_html/debug/'
    normalize = 1
    deta_jj = 1.3
    MJJ = 1200.
    jPt = 100.
    
    NUM_SELECTED_PFCANDS=8
    outfolder = '/ceph/abal/QML/delphes/substructure/CA_decluster'
    
    with h5py.File(filename, "r") as f:
        pf_names=np.array(f['particleFeatureNames'][()],dtype=dt)
        ef_names=np.array(f['eventFeatureNames'][()],dtype=dt)
        dEta=f["eventFeatures"][()][:,9]
        signal_jj=f['eventFeatures'][()]
        signal_pf=f['jetConstituentsList'][()] # Shape: (N,2,100,3)
    sf_names=np.array(['sj1Pt','sj1Eta','sj1Phi','sj1E','sj2Pt','sj2Eta','sj2Phi','sj2E'],dtype=dt)
    
    localfile = filename.split("/")[-1].replace('.h5','')
    print(f'Reading: {localfile}')
    res_type='na'
    
    if 'BROAD' in localfile:
        res_type='br'
    if 'qcd' not in signal:
        sig_mass=re.findall(r'(?<=PU40_)(.*)(?=TeV)',localfile)[0].replace('.','p')
        #sig_folder=os.path.join(outfolder,'grav_'+sig_mass+'_'+res_type) # format: grav_1p5_na
        sig_folder=os.path.join(outfolder,signal+'_'+sig_mass) # format: grav_1p5_na
        if 'grav' in signal:
            sig_folder=os.path.join(outfolder,'grav_'+sig_mass+'_'+res_type)
    else:
        sig_folder=os.path.join(outfolder,signal)
    pathlib.Path(sig_folder).mkdir(parents=True,exist_ok=True)

    outfile = os.path.join(sig_folder,localfile+'.h5')
    print(f'output will be written to {outfile}')
    
    
    signal_mask = (np.abs(dEta) < deta_jj) #& (pt1 > jPt) & (pt2 > jPt) & (mjj > MJJ) & (np.abs(eta1)<2.4) & (np.abs(eta2)<2.4)

    
    signal_jj = signal_jj[signal_mask]

    
    signal_pf=signal_pf[signal_mask]

    #signal_pf_flat=np.reshape(signal_pf,(-1,100,3)) # for jet-wise training and not event-wise
    signal_pf_pxpypzE=np.zeros((signal_pf.shape[0],signal_pf.shape[1],signal_pf.shape[2],4))
    
    signal_pf_pxpypzE[...,0]=signal_pf[...,2]*np.cos(signal_pf[...,1])
    signal_pf_pxpypzE[...,1]=signal_pf[...,2]*np.sin(signal_pf[...,1])
    signal_pf_pxpypzE[...,2]=signal_pf[...,2]*np.sinh(signal_pf[...,0])
    signal_pf_pxpypzE[...,3]=np.sqrt(signal_pf_pxpypzE[...,0]**2+signal_pf_pxpypzE[...,1]**2+signal_pf_pxpypzE[...,2]**2)
    jet_PFCands,evt_subjet_idx, mask, num_PFCands_subleading_jet,evt_subjet_features,evt_subjet_labels=pf.uncluster(signal_pf_pxpypzE,min_pt=jPt,test=False)
    # For jets with more than NUM_SELECTED_PFCANDS//2 PFCands in the subleading jet, we only consider the NUM_SELECTED_PFCANDS//2 leading PFCands
    # If less, then set mask to NUM_SELECTED_PFCANDS - num_PFCands_subleading_jet
    # The jet labels array contains PFCand labels starting from 0 for both leading and subleading jets
    # Something like: [0,1,2,3,4,5,.....,50,0,1,2,......34,999,999,999...] assuming that subjet 1 has 50 PFCands and subjet 2 has 34 PFCands, followed by zero padding to make the shape 100
    # Now applying the mask as evt_subjet_idx<mask_limit[:,None] will allow us to select only the leading NUM_SELECTED_PFCANDS//2 PFCands of each sub-jet, and if the subleading jet
    # has less than NUM_SELECTED_PFCANDS//2 PFCands, then we select the balance PFCands from the leading jet

    #mask_limit=np.where(num_PFCands_subleading_jet>NUM_SELECTED_PFCANDS//2,NUM_SELECTED_PFCANDS//2,NUM_SELECTED_PFCANDS-num_PFCands_subleading_jet)
    
    #pf_mask=evt_subjet_idx<mask_limit[...,None]
    signal_jj=signal_jj[mask]
    signal_truth=np.ones_like(signal_jj[:,0])
    
    with h5py.File(outfile, 'w') as signal_hf:
        signal_hf.create_dataset('particleFeatures', data=jet_PFCands)
        signal_hf.create_dataset('eventFeatures', data=signal_jj)
        signal_hf.create_dataset('particleFeatureNames', data=pf_names)
        signal_hf.create_dataset('eventFeatureNames', data=ef_names)
        signal_hf.create_dataset('truth_label', data=signal_truth)
        signal_hf.create_dataset('truth_label_shape', data=signal_truth.shape)
        signal_hf.create_dataset('num_PFCands_subleading_jet',data=num_PFCands_subleading_jet)
        signal_hf.create_dataset('PFCand_subjet_idx',data=evt_subjet_idx)
        signal_hf.create_dataset('subjet_features',data=evt_subjet_features)
        signal_hf.create_dataset('subjet_feature_names',data=sf_names)
        signal_hf.create_dataset('subjet_labels',data=evt_subjet_labels)
        print("\n\n#### DONE #####\n\n")

if __name__ == '__main__':

    multi=True
    raw_file_dir = '/ceph/bmaier/CASE/delphes/events/'
    file_paths=sorted(glob.glob(raw_file_dir+f'{signal}*/*.h5'))
    
    if multi:
        num_cores=min(len(file_paths), 8)
        print(f"Will use {num_cores} cores to process {len(file_paths)} files")
        pool = Pool(num_cores);sleep(3)
        pool.map(process_signal_region, file_paths) 
    else:
        for filename in file_paths:
            process_signal_region(filename)