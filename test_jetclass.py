import hydra
from omegaconf import DictConfig, OmegaConf
import os
import pathlib
import glob,h5py

# Other imports
import quantum.losses as loss
import numpy as nnp
import helpers.utils as ut
import helpers.path_setter as ps
import case_reader as cr
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import mplhep; mplhep.style.use("CMS")
from sklearn.metrics import roc_curve, roc_auc_score

@hydra.main(config_path="./hydra_configs", config_name="config")
def main(cfg: DictConfig):
    scheme = cfg.scheme
    log_wandb=cfg.log_wandb

    # Set up directories
    save_dir = os.path.join(cfg.save_dir, cfg.seed)
    dump_dir=cfg.dump
    out_dir = os.path.join(cfg.dump, cfg.seed)
    plot_dir = os.path.join(save_dir, 'plots')
    hist_dir = os.path.join(plot_dir, 'jet_histograms')

    pathlib.Path(hist_dir).mkdir(parents=True, exist_ok=True)
    if log_wandb:
        run_id_path = os.path.join(save_dir, "wandb_run_id.txt")
        try:
            assert os.path.exists(run_id_path), "wandb_run_id.txt not found. Ensure training script saved the run ID."
            import wandb
            with open(run_id_path, "r") as f:
                run_id = f.read().strip()

            # Reuse the existing WandB run
            wandb.init(project='1P1Q', id=run_id, resume="must", config=OmegaConf.to_container(cfg))
        except:
            print("WandB run not found. Logging disabled.")
            log_wandb = False
    # Import frozen architecture if available
    try:
        import importlib.util
        qc_spec = importlib.util.spec_from_file_location('qc', os.path.join(save_dir, 'FROZEN_ARCHITECTURE.py'))
        qc = importlib.util.module_from_spec(qc_spec)
        qc_spec.loader.exec_module(qc)
        #import case_reader as cr
        cr_spec = importlib.util.spec_from_file_location('cr', os.path.join(save_dir, 'FROZEN_DATAREADER.py'))
        cr = importlib.util.module_from_spec(cr_spec)
        cr_spec.loader.exec_module(cr)
        print("Successfully imported frozen architecture and dataloaders")
    except ImportError:
        import quantum.architectures as qc
        print("Failure: Frozen architecture not imported. Fetching generic architecture instead")

    # Load arguments and set up quantum autoencoder
    

    sep = cfg.separate_ancilla
    norm_pt = cfg.norm_pt

    qAE = qc.QuantumAutoencoder(wires=cfg.wires, trash_qubits=cfg.trash_qubits, dev_name=cfg.device_name, separate_ancilla=sep, test=True)
    qAE.set_circuit(reuploading=cfg.use_reuploading)
    qc.print_training_params()
    if cfg.loss=='quantum':
        cost_fn = loss.quantum_cost
    else:
        cost_fn = loss.semi_classical_cost
    

    # Load model weights
    try:
        model_path = os.path.join(save_dir, 'trained_model.pickle')
        assert os.path.isfile(model_path), 'Model not found at: ' + model_path
    except:
        print(f"Trained model not found at {model_path}. Will load the most recent checkpoint instead.")
        model_path = sorted(glob.glob(os.path.join(save_dir, 'checkpoints', 'ep*.pickle')))[-1]
    finally:
        qAE.load_weights(model_path)
        print(f"Successfully loaded model at {model_path}")

    # Load test data
    paths = ps.PathSetter(data_path=cfg.data_dir)
    sig_files = sorted(glob.glob(paths.get_data_path(cfg.signal) + '/*.h5'))
    if cfg.dataset.casefold()=='delphes':
        qcd_files = sorted(glob.glob(paths.get_data_path('QCD_SR_test') + '/*.h5'))
        qcd_dataset=cr.CASEDelphesJetDataset(
            filelist=qcd_files,
            input_shape=(len(qc.auto_wires), 3),
            normalize_pt=norm_pt,
            max_samples=cfg.read_n,
            use_subjet_PFCands=cfg.substructure
        )
        sig_dataset=cr.CASEDelphesJetDataset(
            filelist=sig_files,
            input_shape=(len(qc.auto_wires), 3),
            normalize_pt=norm_pt,
            max_samples=cfg.read_n,
            use_subjet_PFCands=cfg.substructure
        )
    elif cfg.dataset.casefold()=='jetclass': 
        qcd_files=sorted(glob.glob(paths.get_data_path('ZJetsToNuNu_flat_test') + '/*.h5'))
        qcd_dataset=cr.CASEJetClassDataset(
            filelist=qcd_files,
            input_shape=(len(qc.auto_wires), 3),
            normalize_pt=norm_pt,
            max_samples=cfg.read_n,
            use_subjet_PFCands=cfg.substructure,selection=cfg.PFCand_selection_type
        )
        sig_dataset=cr.CASEJetClassDataset(
            filelist=sig_files,
            input_shape=(len(qc.auto_wires), 3),
            normalize_pt=norm_pt,
            max_samples=cfg.read_n,
            use_subjet_PFCands=cfg.substructure,selection=cfg.PFCand_selection_type
        )
    else:
        raise NameError(f"Dataset {cfg.dataset} not recognized. Must be either delphes or jetclass")
    
    
    
    if cfg.load:
        with h5py.File(os.path.join(out_dir, 'qcd_results.h5'), 'r') as f:
            qcd_fids = f['fids'][()]
            qcd_costs = f['costs'][()]
            qcd_features = f['jetFeatures'][()]
            qcd_labels = f['truth_label'][()]
            qcd_etaphipt = f['jetConstituentsList'][()]
        with h5py.File(os.path.join(out_dir, cfg.signal, 'sig_results.h5'), 'r') as f:
            sig_fids = f['fids'][()]
            sig_costs = f['costs'][()]
            sig_features = f['jetFeatures'][()]
            sig_labels = f['truth_label'][()]
            sig_etaphipt = f['jetConstituentsList'][()]
    # Inference
    else:
        qcd_etaphipt, qcd_features, qcd_labels = qcd_dataset.load_for_inference()
        sig_etaphipt, sig_features, sig_labels = sig_dataset.load_for_inference()
    
        qcd_costs, qcd_fids = qAE.run_inference(qcd_etaphipt, loss_fn=cost_fn)
        sig_costs, sig_fids = qAE.run_inference(sig_etaphipt, loss_fn=cost_fn)
        
        # Save results
        
        pathlib.Path(os.path.join(out_dir, cfg.signal)).mkdir(parents=True, exist_ok=True)
        with h5py.File(os.path.join(out_dir, 'qcd_results.h5'), 'w') as f:
            f.create_dataset('jetConstituentsList', data=qcd_etaphipt)
            f.create_dataset('jetFeatures', data=qcd_features)
            f.create_dataset('fids', data=qcd_fids)
            f.create_dataset('costs', data=qcd_costs)
            f.create_dataset('truth_label', data=qcd_labels)
        with h5py.File(os.path.join(out_dir, cfg.signal, 'sig_results.h5'), 'w') as f:
            f.create_dataset('jetConstituentsList', data=sig_etaphipt)
            f.create_dataset('jetFeatures', data=sig_features)
            f.create_dataset('fids', data=sig_fids)
            f.create_dataset('costs', data=sig_costs)
            f.create_dataset('truth_label', data=sig_labels)
 
    qcd_fids=qcd_fids
    sig_fids=sig_fids
    qcd_costs=qcd_costs
    sig_costs=sig_costs
    qcd_labels=nnp.zeros(qcd_fids.shape[0])
    sig_labels=nnp.ones(sig_fids.shape[0])
    labels=nnp.concatenate([qcd_labels,sig_labels],axis=0)
    fids=nnp.concatenate([qcd_fids,sig_fids],axis=0)
    costs=nnp.concatenate([qcd_costs,sig_costs],axis=0) 


    #scaler = MinMaxScaler(feature_range=(0, 1.))
    #costs=scaler.fit_transform(costs.reshape(-1,1)).flatten()


    fpr,tpr,thresholds=roc_curve(labels,costs)
    roc_auc=roc_auc_score(labels,costs)

    plot_label=ps.labels[cfg.signal.replace('_flat','')]
    
    bins_qcd,edges_qcd=nnp.histogram(qcd_fids,density=True,bins=50,range=[0,100])
    bins_sig,edges_sig=nnp.histogram(sig_fids,density=True,bins=50,range=[0,100])
    plt.stairs(bins_qcd,edges_qcd,fill=True,label='q/g jets',alpha=0.6)
    plt.stairs(bins_sig,edges_sig,fill=False,label=plot_label)
    plt.minorticks_on()
    #plt.grid(True,which='major',linestyle='--')

    plt.xlabel(r'Quantum Fidelity (%): $\langle T|R \rangle$')
    plt.ylabel('No. of events')
    #plt.yscale('log')
    plt.legend(loc='upper left')
    plt.savefig(os.path.join(plot_dir,f'fidelity_hist_{cfg.signal}.png'))

    plt.clf()
    plt.plot(fpr,tpr,label='AUC = %0.3f' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC: {plot_label}')
    plt.minorticks_on()
    #plt.grid(True,which='major',linestyle='--')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(plot_dir,f'roc_curve_{cfg.signal}.png'))
    plt.clf()
    
    sic=tpr/nnp.sqrt(fpr)
    plt.plot(tpr,sic)
    plt.xlabel('Signal efficiency')
    plt.ylabel('Significance Improvement')
    plt.xlim(0.1,1)
    plt.title('SIC: '+plot_label)
    plt.savefig(os.path.join(plot_dir,f'SIC_{cfg.signal}.png'))
    
    if log_wandb:
        for filename in glob.glob(os.path.join(plot_dir, "*.png")):
            wandb.log({os.path.split(filename)[-1].replace('*.png',''): wandb.Image(filename)})

        # Finish WandB run
        wandb.finish()
if __name__ == "__main__":
    main()
