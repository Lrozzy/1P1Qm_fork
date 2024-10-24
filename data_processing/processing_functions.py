import fastjet as fj
import numpy as np

def bound_angle(theta):
    theta=np.where(theta>np.pi,theta-2*np.pi,theta)
    theta=np.where(theta<-np.pi,theta+2*np.pi,theta)
    return theta

def uncluster(jet_data,min_pt=100,test=False):
    jetdef = fj.JetDefinition(fj.cambridge_algorithm, 1.0)
    nEvts=jet_data.shape[0]
    # If you want to test something new, use up to 20k events to get a general idea
    mask=np.ones(jet_data.shape[0])
    if test: nEvts=min(jet_data.shape[0],10000);mask[nEvts:]=0
    #dcut=0.25
    evt_PFCands=[]
    evt_subjet_idx=[]
    evt_subjet_labels=[]
    num_PFCands=[]
    evt_subjet_features=[]
    print('begin jet clustering and unclustering')
    
    for i in range(nEvts):
        jet_PFCands=[]
        jet_idx=[]
        jet_labels=[]
        nPFCands_softer=[]
        jet_subfeatures=[]
        if (i+1)%1000==0:
            print(f'Unclustering jets from Event {i+1}/{nEvts}')
        j1=jet_data[i,0]
        j2=jet_data[i,1]

        pseudojets1 = [fj.PseudoJet(j1[j,0],j1[j,1],j1[j,2],j1[j,3]) for j in range(j1.shape[0])]
        cluster1 = fj.ClusterSequence(pseudojets1, jetdef)
        jets1 = cluster1.inclusive_jets(ptmin=min_pt)
        
        pseudojets2 = [fj.PseudoJet(j2[j,0],j2[j,1],j2[j,2],j2[j,3]) for j in range(j2.shape[0])]
        cluster2 = fj.ClusterSequence(pseudojets2, jetdef)
        jets2 = cluster2.inclusive_jets(ptmin=min_pt)
        if (len(jets1)!=1) or (len(jets2)!=1): 
            mask[i]=0
            continue
        for jets in [jets1,jets2]:
        #assert len(jets)==1 # we expect only one jet with the given pt cut and radius
            jet_constituents=[]
            subjet_idx=[]
            subjet_labels=[]
            subjet_features=[]
            jets = jets[0]
            if jets.n_exclusive_subjets(0.25)<2:
                exc_subjets = fj.sorted_by_pt(jets.exclusive_subjets(2))
                #import pdb;pdb.set_trace()
            else:
                exc_subjets = fj.sorted_by_pt(jets.exclusive_subjets(0.25))
            
            nPFCands_softer.append(len(exc_subjets[1].constituents()))  
            for j,subjet in enumerate(exc_subjets[:2]): # descending order of pt
                for k,sub_constituent in enumerate(fj.sorted_by_pt(subjet.constituents())):
                    jet_constituents.append(np.array([sub_constituent.eta(),sub_constituent.phi_std(),sub_constituent.pt()]))
                    subjet_idx.append(k)
                    subjet_labels.append(j)
                subjet_features.append(np.array([subjet.pt(),subjet.eta(),subjet.phi_std(),subjet.E()]))        
                        
            jet_constituents=np.array(jet_constituents)
            subjet_features=np.concatenate(subjet_features,axis=0)
            subjet_idx=np.array(subjet_idx)
            subjet_labels=np.array(subjet_labels)
                # zero pad to 100 constituents
            
            if jet_constituents.shape[0]<100:
                jet_constituents=np.concatenate([jet_constituents,np.zeros((100-jet_constituents.shape[0],3))])
                subjet_idx=np.concatenate([subjet_idx,999*np.ones(100-subjet_idx.shape[0])])
                subjet_labels=np.concatenate([subjet_labels,-1*np.ones(100-subjet_labels.shape[0])])
            jet_PFCands.append(jet_constituents);jet_idx.append(subjet_idx);jet_subfeatures.append(subjet_features);jet_labels.append(subjet_labels)
        evt_subjet_features.append(np.stack(jet_subfeatures,axis=0))
        evt_PFCands.append(np.stack(jet_PFCands,axis=0))
        evt_subjet_idx.append(np.stack(jet_idx,axis=0))
        evt_subjet_labels.append(np.stack(jet_labels,axis=0))
        num_PFCands.append(np.array(nPFCands_softer))
    
    return np.array(evt_PFCands),np.array(evt_subjet_idx),np.array(mask).astype(bool),np.array(num_PFCands),np.array(evt_subjet_features),np.array(evt_subjet_labels)


def uncluster_AK4(jet_data,min_pt=50.):
    jetdef = fj.JetDefinition(fj.antikt_algorithm, 0.4)
    nEvts=jet_data.shape[0]
    mask=np.ones(nEvts)
    evt_PFCands=[]
    evt_subjet_idx=[]
    num_PFCands=[]
    print('begin jet clustering and unclustering')
    for i in range(nEvts):
        jet_PFCands=[]
        jet_idx=[]
        nPFCands_softer=[]
        if (i+1)%1000==0:
            print(f'Unclustering jets from Event {i+1}/{nEvts}')
        j1=jet_data[i,0]
        j2=jet_data[i,1]
        
        pseudojets1 = [fj.PseudoJet(j1[j,0],j1[j,1],j1[j,2],j1[j,3]) for j in range(j1.shape[0])]
        cluster1 = fj.ClusterSequence(pseudojets1, jetdef)
        jets1 = cluster1.inclusive_jets(ptmin=min_pt)
        
        pseudojets2 = [fj.PseudoJet(j2[j,0],j2[j,1],j2[j,2],j2[j,3]) for j in range(j2.shape[0])]
        cluster2 = fj.ClusterSequence(pseudojets2, jetdef)
        jets2 = cluster2.inclusive_jets(ptmin=min_pt)
        if (len(jets1)!=1) or (len(jets2)!=1): 
            mask[i]=0
            continue
        for jets in [jets1,jets2]:
        #assert len(jets)==1 # we expect only one jet with the given pt cut and radius
            jet_constituents=[]
            subjet_idx=[]
            subjet_labels=[]
            jets = jets[0]
            exc_subjets = fj.sorted_by_pt(jets.exclusive_subjets(2))
            nPFCands_softer.append(len(exc_subjets[1].constituents()))  
        
            for j,subjet in enumerate(exc_subjets): # descending order of pt
                for k,sub_constituent in enumerate(fj.sorted_by_pt(subjet.constituents())):
                    jet_constituents.append(np.array([sub_constituent.eta(),sub_constituent.phi_std(),sub_constituent.pt()]))
                    subjet_idx.append(k)
                    subjet_labels.append(j)
            jet_constituents=np.array(jet_constituents)
            subjet_idx=np.array(subjet_idx)
            subjet_labels=np.array(subjet_labels)
                # zero pad to 100 constituents
            
            if jet_constituents.shape[0]<100:
                jet_constituents=np.concatenate([jet_constituents,np.zeros((100-jet_constituents.shape[0],3))])
                subjet_idx=np.concatenate([subjet_idx,999*np.ones(100-subjet_idx.shape[0])])
            jet_PFCands.append(jet_constituents);jet_idx.append(subjet_idx)
        evt_PFCands.append(np.stack(jet_PFCands,axis=0))
        evt_subjet_idx.append(np.stack(jet_idx,axis=0))
        num_PFCands.append(np.array(nPFCands_softer))
    
    return np.array(evt_PFCands),np.array(evt_subjet_idx),np.array(mask).astype(bool),np.array(num_PFCands)

