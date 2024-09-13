import h5py
import numpy as np
from re import sub
import awkward as ak
import os
import glob
import pdb
normalize = 1
deta_jj = 1.3
MJJ = 1200.
jPt = 100.
import tqdm,re
import pathlib
import matplotlib;matplotlib.use("Agg")
import matplotlib.pyplot as plt
import fastjet as fj
import awkward as ak

def bound_angle(theta):
    theta=np.where(theta>np.pi,theta-2*np.pi,theta)
    theta=np.where(theta<-np.pi,theta+2*np.pi,theta)
    return theta

def uncluster(jet_data):
    pseudojets = []
    jetdef = fj.JetDefinition(fj.antikt_algorithm, 0.8)
    njets=jet_data.shape[0]//50
    jet_PFCands=[]
    jet_labels=[]
    mask=[]
    nPFCands_softer=[]
    print('begin jet clustering and unclustering')
    for i in range(10000):
        if (i+1)%1000==0:
            print(f'Unclustering jet {i+1}/{njets}')
        j1=jet_data[i]
        pseudojets = [fj.PseudoJet(j1[j,0],j1[j,1],j1[j,2],j1[j,3]) for j in range(j1.shape[0])]
        cluster = fj.ClusterSequence(pseudojets, jetdef)
        jets = cluster.inclusive_jets(ptmin=jPt)
        #assert len(jets)==1 # we expect only one jet with the given pt cut and radius
        jet_constituents=[]
        subjet_label=[]
        
        if len(jets)!=1:
            mask.append(False)
            continue
        
        else:
            mask.append(True)
            jets = jets[0]
            exc_subjets = fj.sorted_by_pt(jets.exclusive_subjets(2))
            nPFCands_softer.append(len(exc_subjets[1].constituents()))  
        
            for j,subjet in enumerate(exc_subjets): # descending order of pt
                for k,sub_constituent in enumerate(fj.sorted_by_pt(subjet.constituents())):
                    jet_constituents.append(np.array([sub_constituent.eta(),sub_constituent.phi(),sub_constituent.pt()]))
                    subjet_label.append(k)
                    
        jet_constituents=np.array(jet_constituents)
        subjet_label=np.array(subjet_label)
            # zero pad to 100 constituents
        
        if jet_constituents.shape[0]<100:
            jet_constituents=np.concatenate([jet_constituents,np.zeros((100-jet_constituents.shape[0],3))])
            subjet_label=np.concatenate([subjet_label,999*np.ones(100-subjet_label.shape[0])])
        jet_PFCands.append(jet_constituents);jet_labels.append(subjet_label)
    return np.array(jet_PFCands),np.array(jet_labels),np.array(mask),np.array(nPFCands_softer)

if __name__ == '__main__':
    dt = h5py.special_dtype(vlen=str)
    ipath='/web/abal/public_html/debug/'

    signal='RSGraviton'
    sig_folder='grav'
    raw_file_dir = '/ceph/bmaier/CASE/delphes/events/'
    file_paths=sorted(glob.glob(raw_file_dir+f'{signal}*/*.h5'))
    outfolder = '/ceph/abal/QML/delphes/signal_sqrtshatTeV_13TeV_PU40_NEW_EXT'
    
    for filename in tqdm.tqdm(file_paths):
        with h5py.File(filename, "r") as f:

            
            
            localfile = filename.split("/")[-1].replace('.h5','')
            print(f'Reading: {localfile}')
            res_type='na'
            
            if 'BROAD' in localfile:
                res_type='br'
            sig_mass=re.findall(r'(?<=PU40_)(.*)(?=TeV)',localfile)[0].replace('.','p')
            sig_folder=os.path.join(outfolder,'grav_'+sig_mass+'_'+res_type) # format: grav_1p5_na
            pathlib.Path(sig_folder).mkdir(parents=True,exist_ok=True)

            outfile = os.path.join(sig_folder,localfile+'.h5')
            print(f'output will be written to {outfile}')
            dEta=f["eventFeatures"][()][:,9]
            
            side_mask = (np.abs(dEta) < deta_jj) #& (pt1 > jPt) & (pt2 > jPt) & (mjj > MJJ) & (np.abs(eta1)<2.4) & (np.abs(eta2)<2.4)

            side_jj=f['eventFeatures'][()]
            side_jj = side_jj[side_mask]

            side_pf=f['jetConstituentsList'][()] # Shape: (N,2,100,3)
            side_pf=side_pf[side_mask]

            side_pf_flat=np.reshape(side_pf,(-1,100,3)) # for jet-wise training and not event-wise
            side_pf_pxpypzE=np.zeros((side_pf_flat.shape[0],100,4))

            side_pf_pxpypzE[:,:,0]=side_pf_flat[:,:,2]*np.cos(side_pf_flat[:,:,1])
            side_pf_pxpypzE[:,:,1]=side_pf_flat[:,:,2]*np.sin(side_pf_flat[:,:,1])
            side_pf_pxpypzE[:,:,2]=side_pf_flat[:,:,2]*np.sinh(side_pf_flat[:,:,0])
            side_pf_pxpypzE[:,:,3]=np.sqrt(side_pf_pxpypzE[:,:,0]**2+side_pf_pxpypzE[:,:,1]**2+side_pf_pxpypzE[:,:,2]**2)
            jet_PFCands,jet_labels, mask, num_PFCands_subleading_jet=uncluster(side_pf_pxpypzE)
            mask_limit=np.where(num_PFCands_subleading_jet>5,5,10-num_PFCands_subleading_jet)
            pf_mask=jet_labels<mask_limit[:,None]
            jet_PFCands_selected=jet_PFCands[pf_mask].reshape(-1,10,3)
            import pdb;pdb.set_trace()
            side_jj=side_jj[mask]
            side_truth=np.zeros_like(side_jj[:,0])
            side_hf = h5py.File(outfile, 'w')
            side_hf.create_dataset('particleFeatures', data=jet_PFCands)
            side_hf.create_dataset('eventFeatures', data=side_jj)
            side_hf.create_dataset('particleFeatureNames', data=np.array(f['particleFeatureNames'][()],dtype=dt))
            side_hf.create_dataset('eventFeatureNames', data=np.array(f['eventFeatureNames'][()],dtype=dt))
            side_hf.create_dataset('truth_label', data=jet_labels)
            side_hf.create_dataset('truth_label_shape', data=jet_labels.shape)
            side_hf.create_dataset('num_PFCands_subleading_jet',data=num_PFCands_subleading_jet)
            side_hf.close()
