path='./network_runs/MET_b1pt_lep1pt_lep2pt/train_100/run_3/'
from utils import Unpickle,print_events
import numpy as np
import sklearn.metrics as skm

test_dict=Unpickle("test",path=path)
print_events(test_dict)
bg=test_dict.pop('sample_bg_morestat.csv')
for key,val in test_dict.items():
    if type(val)!=np.ndarray: continue
    auc=skm.roc_auc_score(np.concatenate((np.zeros(len(bg)),np.ones(len(val))) ),
                          np.concatenate((1-bg,1-val))
                          )
    print ('Signal: ',key.split('_')[1],'AUC:',auc)
