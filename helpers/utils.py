import numpy as np

def getIndex(which:str='particle',feat:str=None)->int:
    if which=='particle':
        nameArray=particleFeatureNames
    elif which=='event':
        nameArray=eventFeatureNames
    else:
        print("arg which must be either event or particle. Returning -1 as index")
        return -1
    try:
        idx=nameArray.index(feat)
    except ValueError as v:
        print(f"item {feat} not found in list {nameArray}")
        idx=-1
    return -1


path_dict:dict[str:str]={'QCD_train':'/storage/9/abal/CASE/delphes/qcd_sqrtshatTeV_13TeV_PU40_NEW_EXT_sideband_parts/train/',
           'QCD_test':'/storage/9/abal/CASE/delphes/qcd_sqrtshatTeV_13TeV_PU40_NEW_EXT_sideband_parts/test/'
           }

eventFeatureNames:list[str]=['mJJ', 'j1Pt', 'j1Eta', 'j1Phi', 'j1M', 'j1E', 'j2Pt',
       'j2M', 'j2E', 'DeltaEtaJJ', 'DeltaPhiJJ']

particleFeatureNames:list[str]=['eta', 'phi', 'pt']

