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
    return idx


path_dict:dict[str:str]={'QCD_train':'/storage/9/abal/CASE/delphes/qcd_sqrtshatTeV_13TeV_PU40_NEW_EXT_sideband_parts/train/',
        'QCD_test':'/storage/9/abal/CASE/delphes/qcd_sqrtshatTeV_13TeV_PU40_NEW_EXT_sideband_parts/test/',
        'QCD_SR':'/storage/9/abal/CASE/delphes/qcd_sqrtshatTeV_13TeV_PU40_NEW_EXT_signalregion_parts/',
        'QAE_save':'/work/abal/qae_hep/saved_models/',
        'QCD_lib':'/storage/9/abal/CASE/delphes/qcd_sqrtshatTeV_13TeV_PU40_NEW_EXT_sideband_parts/library/',
        'QAE_dump':'/ceph/abal/QML/dumps/',
        'grav_1p5_narrow':'/storage/9/abal/CASE/delphes/grav_1p5_na/',
        'grav_2p5_narrow':'/storage/9/abal/CASE/delphes/grav_2p5_na/',
        'grav_3p5_narrow':'/storage/9/abal/CASE/delphes/grav_3p5_na/',
        'grav_4p5_narrow':'/storage/9/abal/CASE/delphes/grav_4p5_na/',
        'grav_3p5_broad':'/storage/9/abal/CASE/delphes/grav_3p5_br/',
        'grav_2p5_broad':'/storage/9/abal/CASE/delphes/grav_2p5_br/',
        'grav_1p5_broad':'/storage/9/abal/CASE/delphes/grav_1p5_br/',
        'AtoHZ_1p5':'/storage/9/abal/CASE/delphes/AtoHZ_1p5/',
        'AtoHZ_2p5':'/storage/9/abal/CASE/delphes/AtoHZ_2p5/',
        'AtoHZ_3p5':'/storage/9/abal/CASE/delphes/AtoHZ_3p5/',
        'AtoHZ_4p5':'/storage/9/abal/CASE/delphes/AtoHZ_4p5/',
           }

eventFeatureNames:list[str]=['mJJ', 'j1Pt', 'j1Eta', 'j1Phi', 'j1M', 'j1E', 'j2Pt',
       'j2M', 'j2E', 'DeltaEtaJJ', 'DeltaPhiJJ']

particleFeatureNames:list[str]=['eta', 'phi', 'pt']

labels={'grav_1p5_narrow':'$M_{grav}=1.5$ TeV','grav_2p5_narrow':'$M_{grav}=2.5$ TeV','grav_3p5_narrow':'$M_{grav}=3.5$ TeV'\
        ,'grav_4p5_narrow':'$M_{grav}=4.5$ TeV','AtoHZ_1p5':'$M_{A}=1.5$ TeV','AtoHZ_2p5':'$M_{A}=2.5$ TeV',\
            'AtoHZ_3p5':'$M_{A}=3.5$ TeV','AtoHZ_4p5':'$M_{A}=4.5$ TeV'}