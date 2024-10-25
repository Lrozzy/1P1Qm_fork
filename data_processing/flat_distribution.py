import os,glob,h5py,pathlib
import numpy as np
import tqdm
from sklearn.utils import resample
import matplotlib.pyplot as plt

def flatten_feature_distribution(feature, num_events=10):

    # Create bins for the feature distribution
    min_feature = 1100.
    max_feature = 7000.
    #bins = [1126,1181,1246,1313,1383,1455,1530,1607,1687,1770,1856,1945,2037,2132,2231,2332,2438,2546,2659,2775,2895,3019,3147,3279,3416,3558,3704,3854,4010,4171,4337,4509,4686,4869,5058,5253,5500,5663,5877,6099,6328,6564,6808,8000]
    bins=np.arange(min_feature,max_feature,50)
    #print("Bins: ",bins)

    #bins=[1000,1700,2400,3100,3800,4500,5200,5900,6600,7300,8000]
    num_bins=len(bins)-1
    # Find which bin each feature value falls into
    bin_indices = np.digitize(feature, bins)

    # Sample an approximately equal number of events from each bin
    sampled_indices = []

    for b in range(1, len(bins)):
        # Get indices of events in the current bin
        bin_mask = bin_indices == b
        bin_events = np.where(bin_mask)[0]

        # Sample from this bin, proportional to bin population
        if len(bin_events) > 0:
            n_samples = min(len(bin_events), num_events)
            sampled_bin_events = resample(bin_events, replace=False, n_samples=n_samples)
            sampled_indices.extend(sampled_bin_events)

        # If necessary, adjust to exactly 30k events
        #sampled_indices = resample(sampled_indices, replace=False, n_samples=30000)

        # Access other event features using the sampled indices
        

    return sampled_indices
def sample_from_feature(feature,num_events=30000):
    probs=np.exp(feature/1000)
    probs=probs/np.sum(probs)
    sampled_indices=np.random.choice(np.arange(feature.shape[0]),size=num_events,replace=False,p=probs)
    return sampled_indices

if __name__ == '__main__':
    #base_dir='/storage/9/abal/CASE/delphes/qcd_sqrtshatTeV_13TeV_PU40_NEW_EXT_sideband_parts/train/'
    #out_dir='/storage/9/abal/CASE/delphes/qcd_sqrtshatTeV_13TeV_PU40_NEW_EXT_sideband_parts/flat_train/'
    base_dir='/ceph/abal/QML/delphes/substructure/CA_decluster/qcd_sqrtshatTeV_13TeV_PU40_NEW_EXT_sideband_parts/train/'
    out_dir='/ceph/abal/QML/delphes/substructure/CA_decluster/qcd_sqrtshatTeV_13TeV_PU40_NEW_EXT_sideband_parts/flat_train/'
    pathlib.Path(out_dir).mkdir(parents=True,exist_ok=True)

    file_paths=glob.glob(os.path.join(base_dir,'*.h5'))
    j1pt_idx=1
    j2pt_idx=6
    mjj_idx=0
    pfc=[]
    eventFeatures=[]
    PFCand_subjet_idx=[]
    num_PFCands_subleading_jet=[]
    subjet_features=[]
    subjet_labels=[]
    for i,file_path in enumerate(tqdm.tqdm(file_paths)):
        with h5py.File(file_path,'r') as f:
            feature=f['eventFeatures'][:,mjj_idx]
            sampled_indices=flatten_feature_distribution(feature)#sample_from_feature(feature,num_events=1000)#
            pfc.append(f['jetConstituentsList'][()][sampled_indices])
            eventFeatures.append(f['eventFeatures'][()][sampled_indices])
            PFCand_subjet_idx.append(f['PFCand_subjet_idx'][()][sampled_indices])
            num_PFCands_subleading_jet.append(f['num_PFCands_subleading_jet'][()][sampled_indices])
            subjet_features.append(f['subjet_features'][()][sampled_indices])
            subjet_labels.append(f['subjet_labels'][()][sampled_indices])

            if i==0:
                eventFeatureNames=f['eventFeatureNames'][()]
                jetConstituentNames=f['particleFeatureNames'][()]
                subjet_feature_names=f['subjet_feature_names'][()]
    pfc=np.concatenate(pfc,axis=0)
    eventFeatures=np.concatenate(eventFeatures,axis=0)
    PFCand_subjet_idx=np.concatenate(PFCand_subjet_idx,axis=0)
    num_PFCands_subleading_jet=np.concatenate(num_PFCands_subleading_jet,axis=0)
    subjet_features=np.concatenate(subjet_features,axis=0)
    subjet_labels=np.concatenate(subjet_labels,axis=0)

    import pdb;pdb.set_trace()
    with h5py.File(os.path.join(out_dir,'flat_mjj_sample.h5'),'w') as f:
        f.create_dataset('jetConstituentsList',data=pfc)
        f.create_dataset('eventFeatures',data=eventFeatures)
        f.create_dataset('eventFeatureNames',data=eventFeatureNames)
        f.create_dataset('jetConstituentNames',data=jetConstituentNames)
        f.create_dataset('truth_labels',data=np.zeros(eventFeatures.shape[0]))
        f.create_dataset('subjet_features',data=subjet_features)
        f.create_dataset('subjet_feature_names',data=subjet_feature_names)
        f.create_dataset('subjet_labels',data=subjet_labels)
        f.create_dataset('num_PFCands_subleading_jet',data=num_PFCands_subleading_jet)
        f.create_dataset('PFCand_subjet_idx',data=PFCand_subjet_idx)