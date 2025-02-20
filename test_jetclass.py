import hydra
from omegaconf import DictConfig, OmegaConf
import os
import pathlib
import glob,h5py,sys

# Other imports
import numpy as nnp
import helpers.utils as ut
import helpers.path_setter as ps
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import mplhep; mplhep.style.use("CMS")
from sklearn.metrics import roc_curve, roc_auc_score,precision_recall_curve

@hydra.main(config_path="./hydra_configs", config_name="config")
def main(cfg: DictConfig):
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
        loss_spec = importlib.util.spec_from_file_location('loss', os.path.join(save_dir, 'FROZEN_LOSS.py'))
        loss = importlib.util.module_from_spec(loss_spec)
        loss_spec.loader.exec_module(loss)
        print("Successfully imported frozen architecture and dataloaders")
    except ImportError as e:
        print(e)
        import quantum.architectures as qc
        import case_reader as cr
        import quantum.losses as loss
        print("Failure: Frozen architecture not imported. Fetching generic architecture instead")
        sys.exit(0)
        
    # Load arguments and set up quantum autoencoder
    
        

    norm_pt = cfg.norm_pt

    qClassifier = qc.QuantumClassifier(wires=cfg.wires, dev_name=cfg.device_name, test=True,layers=cfg.num_layers)
    qClassifier.set_circuit()
    qc.print_training_params()
    cost_fn=loss.batched_VQC_cost
    
    # Load model weights
    try:
        model_path = os.path.join(save_dir, 'trained_model.pickle')
        assert os.path.isfile(model_path), 'Model not found at: ' + model_path
    except:
        #history=ut.Unpickle(os.path.join(save_dir, 'history.pickle'))
        #val_auc=history['auc']
        #best_model=sorted(glob.glob(os.path.join(save_dir, 'checkpoints', 'ep*.pickle')))[nnp.argmax(val_auc)-1]
        #print(f"Trained model not found at {model_path}. Will load the most recent checkpoint instead.")
        model_path = sorted(glob.glob(os.path.join(save_dir, 'checkpoints', 'ep*.pickle')))[-1]
        print(f"Trained model not found at {model_path}. Will load the best model")
        #model_path = best_model
    
    if cfg.load_best:
        history=ut.Unpickle(os.path.join(save_dir, 'history.pickle'))
        val_auc=history['auc']
        best_model=sorted(glob.glob(os.path.join(save_dir, 'checkpoints', 'ep*.pickle')))[nnp.argmax(val_auc)-1]
        model_path = best_model
    qClassifier.load_weights(model_path)
     
    print(f"Successfully loaded model at {model_path}")

    # Load test data
    paths = ps.PathSetter(data_path=cfg.data_dir)
        
    dataset=cr.CASEJetClassDataset(
        filelist=sorted(glob.glob(paths.get_data_path('VQC_test') + '/*.h5')),
        input_shape=(len(qc.auto_wires), 3),
        normalize_pt=norm_pt,
        max_samples=cfg.read_n)
    
    if cfg.load:
        with h5py.File(os.path.join(out_dir, 'qcd_results.h5'), 'r') as f:
            scores = f['scores'][()]
            costs = f['costs'][()]
            features = f['jetFeatures'][()]
            labels = f['truth_labels'][()]
            etaphipt = f['jetConstituentsList'][()]
        
    # Inference
    else:
        etaphipt, features, labels = dataset.load_for_inference()
        #   import pdb;pdb.set_trace()
        costs, scores = qClassifier.run_inference(data=etaphipt, labels=labels,loss_fn=cost_fn,loss_type=cfg.loss)
        
        # Save results
        
        pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
        with h5py.File(os.path.join(out_dir, 'qcd_results.h5'), 'w') as f:
            f.create_dataset('jetConstituentsList', data=etaphipt)
            f.create_dataset('jetFeatures', data=features)
            f.create_dataset('scores', data=scores)
            f.create_dataset('costs', data=costs)
            f.create_dataset('truth_labels', data=labels)
        
    fpr,tpr,thresholds=roc_curve(labels,scores)
    roc_auc=roc_auc_score(labels,scores)
    print(f'AUC={roc_auc:.3f}')
    plot_label=r'$t \rightarrow bq\overline{q}$'
    pathlib.Path(os.path.join(plot_dir,'ROC_data')).mkdir(parents=True, exist_ok=True)
    npz_path = os.path.join(plot_dir, 'ROC_data', 'FPR_TPR.npz')
    nnp.savez(npz_path, fpr=fpr, tpr=tpr,auc=roc_auc,thresholds=thresholds)
    print(f"ROC curve data saved to {npz_path}")
    
    bins_qcd,edges_qcd=nnp.histogram(scores[labels==0],density=True,bins=50,range=[0,2])
    bins_sig,edges_sig=nnp.histogram(scores[labels==1],density=True,bins=50,range=[0,2])
    plt.stairs(bins_qcd,edges_qcd,fill=True,label='q/g jets',alpha=0.6)
    plt.stairs(bins_sig,edges_sig,fill=False,label=plot_label)
    plt.minorticks_on()
    #plt.grid(True,which='major',linestyle='--')

    plt.xlabel('Classifier Score')
    plt.ylabel('No. of events')
    #plt.yscale('log')
    plt.legend(loc='upper left')
    plt.savefig(os.path.join(plot_dir,f'classifier_score_hist.png'))

    plt.clf()
    plt.plot(fpr,tpr,label='AUC = %0.3f' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC: {plot_label} vs q/g jets')
    plt.minorticks_on()
    #plt.grid(True,which='major',linestyle='--')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(plot_dir,f'roc_curve.png'))
    plt.clf()
    sic=tpr/nnp.sqrt(fpr)
    plt.plot(tpr,sic)
    plt.xlabel('Signal efficiency')
    plt.ylabel('Significance Improvement')
    plt.xlim(0.1,1)
    plt.title(f'SIC: {plot_label} vs q/g jets')
    plt.savefig(os.path.join(plot_dir,f'SIC.png'))
    plt.clf()
    plt.plot(tpr,1./fpr,label='Rej$_{X}$')
    plt.xlabel('TPR')
    plt.ylabel('1/FPR')
    plt.yscale('log')
    plt.title(f'Rejection vs Signal Efficiency: {plot_label} vs q/g jets')
    plt.legend()
    plt.savefig(os.path.join(plot_dir,f'rejection.png'))
    if log_wandb:
        for filename in glob.glob(os.path.join(plot_dir, "*.png")):
            wandb.log({os.path.split(filename)[-1].replace('*.png',''): wandb.Image(filename)})
        wandb.log({'test_AUC':roc_auc})
        wandb.finish()

if __name__ == "__main__":
    main()
