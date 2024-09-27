from argparse import ArgumentParser
import os,pathlib,glob
from numthreads import omp_num_threads

parser=ArgumentParser(description='select options to train quantum autoencoder')
parser.add_argument('--seed',default=9999,type=int,help='Some number to index the run')
parser.add_argument('--read_n',default=10000,type=int,help='No. of test events to read in')
parser.add_argument('--epoch_n',default=1,type=int,help='If you want to load some checkpoint at epoch N')
parser.add_argument('--dump',default=False,action='store_true')
parser.add_argument('--load',default=False,action='store_true')
parser.add_argument('--backend',default='autograd')
parser.add_argument('--n_threads',default='1',type=str)
parser.add_argument('--signal',default='grav_2p5_narrow',help='Signal to test against')
parser.add_argument('--device_name',default='default.qubit',help='device name for the quantum circuit. If you use lightning.kokkos, be sure \
    to set the OMP_PROC_BIND and OMP_NUM_THREADS environment variables')
args=parser.parse_args()
# if args.device_name=='lightning.kokkos':
#     os.environ['OMP_NUM_THREADS']=args.n_threads
#     os.environ['OMP_PROC_BIND']='true'
#     print(f"Initialized device {args.device_name} with {os.environ['OMP_NUM_THREADS']} threads")
# else:
#     print(f"Initialized device {args.device_name}")

with omp_num_threads(int(args.n_threads)):
    import quantum.architectures as qc
    import quantum.losses as loss
    import numpy as nnp
    import helpers.utils as ut
    import helpers.path_setter as ps
    import case_reader as cr
    import matplotlib.pyplot as plt
    import matplotlib;matplotlib.use('Agg')
    from sklearn.metrics import roc_curve,roc_auc_score
    from sklearn.preprocessing import MinMaxScaler


    read_n=args.read_n

    dump_dir=os.path.join(ps.path_dict['QAE_dump'],str(args.seed))
    pathlib.Path(dump_dir).mkdir(parents=True,exist_ok=True)
    save_dir=os.path.join(ps.path_dict['QAE_save'],str(args.seed))
    plot_dir=os.path.join(save_dir,'plots')
    hist_dir=os.path.join(plot_dir,'mjj_histograms')
    pathlib.Path(hist_dir).mkdir(parents=True,exist_ok=True)

    assert os.path.isfile(os.path.join(save_dir,'args.pickle')),'args.pickle not found in: '+save_dir

    backend_name=args.backend
    device_name=args.device_name
    test_args=ut.Unpickle(os.path.join(save_dir,'args.pickle'))
    qAE=qc.QuantumAutoencoder(wires=test_args.wires, trash_qubits=test_args.trash_qubits,dev_name=args.device_name,test=True)
    qAE.set_circuit(reuploading=True)
    qc.print_training_params()

    semi_classical_cost=loss.semi_classical_cost

    try:
        model_path=os.path.join(save_dir,'trained_model.pickle')
        assert os.path.isfile(model_path),'Model not found at: '+model_path
    except:
        print(f"Trained model not found at {model_path} \n Will load most recent checkpoint instead")
        model_path=sorted(glob.glob(os.path.join(save_dir,'checkpoints','ep*.pickle')))[-1]
    finally:
        qAE.load_weights(model_path)
        print(f"Successfully loaded model at {model_path}")
        
    try:
        history=ut.Unpickle(path=os.path.join(save_dir,'history.pickle'))
        val_loss=history['val']
        epoch=nnp.argmin(val_loss)
        print (epoch,val_loss[epoch])
    except:
        print("Did not find history file. Skipping, its not important except for statistics.")
    fid_dict={}



    # test_loader=cr.CASEDelphesDataLoader(filelist=sorted(glob.glob(ps.path_dict['QCD_lib']+'/*.h5')),batch_size=1000,\
    #                                      input_shape=(len(qc.auto_wires),3),train=False,max_samples=read_n,which=which)  # Shuffle is set to False
    # sig_loader=cr.CASEDelphesDataLoader(filelist=sorted(glob.glob(ps.path_dict['grav_4p5_narrow']+'/*.h5')),batch_size=1000,\
    #                                     input_shape=(len(qc.auto_wires),3),train=False,max_samples=read_n,which=which)  # Shuffle is set to False
    pathlib.Path(os.path.join(dump_dir,args.signal)).mkdir(parents=True,exist_ok=True)

    if args.load:
        qcd_fids_j1=nnp.load(os.path.join(dump_dir,'qcd_fids_j1.npy'))
        sig_fids_j1=nnp.load(os.path.join(dump_dir,args.signal,'sig_fids_j1.npy'))
        qcd_fids_j2=nnp.load(os.path.join(dump_dir,'qcd_fids_j2.npy'))
        sig_fids_j2=nnp.load(os.path.join(dump_dir,args.signal,'sig_fids_j2.npy'))
        qcd_costs_j1=nnp.load(os.path.join(dump_dir,'qcd_costs_j1.npy'))
        sig_costs_j1=nnp.load(os.path.join(dump_dir,args.signal,'sig_costs_j1.npy'))
        qcd_costs_j2=nnp.load(os.path.join(dump_dir,'qcd_costs_j2.npy'))
        sig_costs_j2=nnp.load(os.path.join(dump_dir,args.signal,'sig_costs_j2.npy'))
        qcd_mjj=nnp.load(os.path.join(dump_dir,'qcd_mjj.npy'))
        sig_mjj=nnp.load(os.path.join(dump_dir,args.signal,'sig_mjj.npy'))
        qcd_labels=nnp.zeros(qcd_fids_j1.shape[0])
        sig_labels=nnp.ones(sig_fids_j1.shape[0])
    else:
        qcd_j1_etaphipt,qcd_j2_etaphipt,qcd_mjj,qcd_labels=cr.CASEDelphesJetDataset(filelist=sorted(glob.glob(ps.path_dict['QCD_SR']+'/*.h5')),\
                                                                                input_shape=(len(qc.auto_wires),3),max_samples=read_n).load_for_inference()
        sig_j1_etaphipt,sig_j2_etaphipt,sig_mjj,sig_labels=cr.CASEDelphesJetDataset(filelist=sorted(glob.glob(ps.path_dict[args.signal]+'/*.h5')),\
                                                                                input_shape=(len(qc.auto_wires),3),max_samples=10000).load_for_inference()
        qcd_j1_etaphipt,qcd_j2_etaphipt,sig_j1_etaphipt,sig_j2_etaphipt=cr.rescale_and_reshape([qcd_j1_etaphipt,qcd_j2_etaphipt,sig_j1_etaphipt,sig_j2_etaphipt])
        qcd_costs_j1,qcd_fids_j1=qAE.run_inference(qcd_j1_etaphipt,loss_fn=semi_classical_cost)
        sig_costs_j1,sig_fids_j1=qAE.run_inference(sig_j1_etaphipt,loss_fn=semi_classical_cost)
        qcd_costs_j2,qcd_fids_j2=qAE.run_inference(qcd_j2_etaphipt,loss_fn=semi_classical_cost)
        sig_costs_j2,sig_fids_j2=qAE.run_inference(sig_j2_etaphipt,loss_fn=semi_classical_cost)
        
    if args.dump:
        
        nnp.save(os.path.join(dump_dir,'qcd_fids_j1.npy'),qcd_fids_j1)
        nnp.save(os.path.join(dump_dir,args.signal,'sig_fids_j1.npy'),sig_fids_j1)
        nnp.save(os.path.join(dump_dir,'qcd_fids_j2.npy'),qcd_fids_j2)
        nnp.save(os.path.join(dump_dir,args.signal,'sig_fids_j2.npy'),sig_fids_j2)
        nnp.save(os.path.join(dump_dir,'qcd_costs_j1.npy'),qcd_costs_j1)
        nnp.save(os.path.join(dump_dir,args.signal,'sig_costs_j1.npy'),sig_costs_j1)
        nnp.save(os.path.join(dump_dir,'qcd_costs_j2.npy'),qcd_costs_j2)
        nnp.save(os.path.join(dump_dir,args.signal,'sig_costs_j2.npy'),sig_costs_j2)
        nnp.save(os.path.join(dump_dir,'qcd_mjj.npy'),qcd_mjj)
        nnp.save(os.path.join(dump_dir,args.signal,'sig_mjj.npy'),sig_mjj)
        

    import pdb;pdb.set_trace()
    qcd_fids=nnp.maximum(qcd_fids_j1,qcd_fids_j2)
    sig_fids=nnp.maximum(sig_fids_j1,sig_fids_j2)
    qcd_costs=nnp.minimum(qcd_costs_j1,qcd_costs_j2)
    sig_costs=nnp.minimum(sig_costs_j1,sig_costs_j2)
    qcd_labels=nnp.zeros(qcd_fids.shape[0])
    sig_labels=nnp.ones(sig_fids.shape[0])
    labels=nnp.concatenate([qcd_labels,sig_labels],axis=0)
    fids=nnp.concatenate([qcd_fids,sig_fids],axis=0)
    costs=nnp.concatenate([qcd_costs,sig_costs],axis=0) 
    mjj=nnp.concatenate([qcd_mjj,sig_mjj],axis=0)
    #import pdb;pdb.set_trace()
    scaler = MinMaxScaler(feature_range=(0, 1.))
    costs=scaler.fit_transform(costs.reshape(-1,1)).flatten()

    #import pdb;pdb.set_trace()
    fpr,tpr,thresholds=roc_curve(labels,costs)
    roc_auc=roc_auc_score(labels,costs)


    bins_qcd,edges_qcd=nnp.histogram(qcd_fids,density=True,bins=200,range=[0,100])
    bins_sig,edges_sig=nnp.histogram(sig_fids,density=True,bins=200,range=[0,100])
    plt.stairs(bins_qcd,edges_qcd,fill=True,label='QCD',alpha=0.6)
    plt.stairs(bins_sig,edges_sig,fill=False,label=ps.labels[args.signal])
    plt.minorticks_on()
    plt.grid(True,which='major',linestyle='--')

    plt.xlabel(u'Quantum Fidelity (%): \u27e8$T|R$\u27e9')
    plt.ylabel('No. of events')
    plt.yscale('log')
    plt.legend(loc='upper left')
    plt.savefig(os.path.join(plot_dir,f'fidelity_hist_{args.signal}.png'))

    plt.clf()
    plt.plot(fpr,tpr,label='AUC = %0.3f' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC: {ps.labels[args.signal]}')
    plt.minorticks_on()
    plt.grid(True,which='major',linestyle='--')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(plot_dir,f'roc_curve_{args.signal}.png'))

    plt.clf()
    plt.scatter(mjj[labels==0],costs[labels==0],label='QCD',s=2,marker='o', alpha=.6,)
    plt.scatter(mjj[labels==1],costs[labels==1],label=ps.labels[args.signal],s=1,marker='s', alpha=.4,)
    #plt.hist(costs[labels==0],bins=100,range=[0,1],label='QCD',histtype='step',linewidth=2)
    #plt.hist(costs[labels==1],bins=100,range=[0,1],label=ps.labels[args.signal],histtype='step',linewidth=2)
    plt.legend(loc='upper left')
    plt.minorticks_on()
    plt.grid(True,which='major',linestyle='--')
    plt.xlabel('$m_{JJ}$ (GeV)')
    plt.ylabel('Cost = $1-$ \u27e8$T|R$\u27e9 (scaled to $[0,1]$)')
    plt.title('Signal: '+ps.labels[args.signal])
    #plt.ylabel('No. of events')
    #plt.xlabel('Cost = $1-$ \u27e8$T|R$\u27e9 (scaled to $[0,1]$)')

    plt.savefig(os.path.join(plot_dir,f'cost_vs_mjj_{args.signal}.png'))
    plt.clf()
    effs=nnp.arange(10,100,0.5)
    sig_cuts=nnp.percentile(sig_costs,effs)
    sig_cuts,idx=nnp.unique(sig_cuts,return_index=True)
    effs=effs[idx]
    significance=[]
    plot_eff=[]
    
    for cut in sig_cuts:
        sig_eff=nnp.sum(sig_costs>=cut)/sig_costs.shape[0]
        qcd_eff=nnp.sum(qcd_costs>=cut)/qcd_costs.shape[0]
        if sig_eff<0.15: continue
        if qcd_eff>0:
            plot_eff.append(sig_eff)
            significance.append(sig_eff/nnp.sqrt(qcd_eff))
    sig_cuts=nnp.percentile(sig_costs,effs)
    #import pdb;pdb.set_trace()
    plt.plot(plot_eff,significance)
    plt.minorticks_on()
    plt.grid(True,which='major',linestyle='--')
    plt.xlabel('Signal efficiency')
    plt.ylabel('Significance Improvement')
    plt.xlim(0.1,1)
    plt.title('SIC: '+ps.labels[args.signal])
    plt.savefig(os.path.join(plot_dir,f'SIC_{args.signal}.png'))

    plt.clf()
    effs=nnp.array([30,60,70,80,90,100])
    inv_effs=100-effs
    cuts=nnp.percentile(costs,effs)
    plt.hist(mjj[labels==0],bins=100,range=[1000,5000],label='QCD inclusive',histtype='step',linewidth=1.,density=True)
    for i,(lower_cut,upper_cut) in enumerate(zip(cuts[:-1],cuts[1:])):
        
        mask=(costs>=lower_cut)&(costs<upper_cut)
        cat_labels=labels[mask]
        cat_mjj=mjj[mask]
        if inv_effs[i+1]==100: leg=f'>Q{inv_effs[i]}'
        else: leg=f'Q{inv_effs[i]}-{inv_effs[i+1]}'
        plt.hist(cat_mjj[cat_labels==0],bins=100,range=[1000,5000],label=f'QCD: {leg}',histtype='step',linewidth=1.,density=True,alpha=0.8)
        #plt.hist(cat_mjj[cat_labels==1],bins=100,range=[1000,5000],label=ps.labels[args.signal],histtype='step',linewidth=2)
    plt.legend(loc='upper right')
    # if inv_effs[i]==50.: 
    #     import pdb;pdb.set_trace()
    plt.minorticks_on()
    plt.grid(True,which='major',linestyle='--')
    plt.xlabel('$m_{JJ}$ (GeV)')
    plt.ylabel('No. of events')
    plt.title("Quantum VAE")
    plt.savefig(os.path.join(hist_dir,f'mjj_cut_{args.signal}.png'))
    plt.clf()
        