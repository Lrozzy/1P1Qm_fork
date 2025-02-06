import hydra
from omegaconf import DictConfig,OmegaConf
import os
import pathlib
import subprocess
import datetime
import glob
import time
import matplotlib.pyplot as plt
import helpers.utils as ut
import case_reader as cr
import helpers.path_setter as ps
import quantum.losses as loss
from loguru import logger
import wandb

@hydra.main(config_path="./hydra_configs/", config_name="config")
def main(cfg: DictConfig):
    # Set up directories
    base_dir:str=cfg.base_dir
    save_dir = os.path.join(cfg.save_dir, cfg.seed)
    plot_dir = os.path.join(save_dir, 'plots')
    pathlib.Path(plot_dir).mkdir(parents=True, exist_ok=True)

    # Initialize WandB
    try:
        run_str=f"{os.getlogin()}_{cfg.seed}"
    except:
        run_str=f"abal_{cfg.seed}"
    
    wandb.init(project="1P1Q", config=OmegaConf.to_container(cfg), name=run_str,notes=cfg.desc)
    with open(os.path.join(save_dir, "wandb_run_id.txt"), "w") as f:
        f.write(wandb.run.id)

    # Logging setup
    logger.add(os.path.join(save_dir, 'logs.log'), rotation='10 MB', backtrace=True, diagnose=True, level='DEBUG', mode="w")
    logger.info("########################################### \n\n")
    logger.info(f"This circuit contains {cfg.wires} qubits")
    
    print("Will save models to: ", save_dir)

    # Save initial arguments for logging purposes
    ut.Pickle(cfg, 'args', path=save_dir)
    with open(os.path.join(save_dir, 'args.txt'), 'w+') as f:
        f.write(repr(cfg))

    # Further setup based on config
    
    if cfg.resume:
        test_args = ut.Unpickle(os.path.join(save_dir, 'args.pickle'))
        import importlib; qc = importlib.import_module('saved_models.' + cfg.seed + '.FROZEN_ARCHITECTURE')
        model_path = sorted(glob.glob(os.path.join(save_dir, 'checkpoints', 'ep*.pickle')))[-1]
        init_weights = ut.Unpickle(model_path)

        logger.add(os.path.join(cfg.save_dir, 'logs.log'), rotation='10 MB', backtrace=True, diagnose=True, level='DEBUG', mode="a")
        logger.info("########################################### \n\n")
        logger.info(f"Resuming training from last checkpoint at {model_path}")
        logger.info(f"Using arguments specified in original training run")
        logger.info(f"Training resumed at {datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}")
        logger.info("Current weights are: ", init_weights)
        logger.info("\n\n ########################################### \n\n")
        cfg = test_args
        cfg.seed = str(cfg.seed)
    else:
        import quantum.architectures as qc
        logger.info("########################################### \n\n")
        logger.info(f"This circuit contains {cfg.wires} qubits")
        logger.info("\n\n ########################################### \n\n")
        tmpfile = os.path.join(save_dir, 'FROZEN_ARCHITECTURE.py')
        subprocess.run(['cp', os.path.join(base_dir,'quantum/architectures.py') ,tmpfile])
        tmpfile = os.path.join(save_dir, 'FROZEN_DATAREADER.py')
        subprocess.run(['cp', os.path.join(base_dir,'case_reader.py'), tmpfile])
        tmpfile = os.path.join(save_dir, 'FROZEN_LOSS.py')
        subprocess.run(['cp', os.path.join(base_dir,'quantum/losses.py'), tmpfile])
        
        
    logger.info(f"Feature are scaled to the following limits: {ut.feature_limits}")

    if cfg.norm_pt:
        logger.info(f"pT will not be scaled to the above limit. Will be normalized using 1/jet_pt")
    else:
        logger.info(f"pT will also be scaled assuming above maxima")
    if cfg.flat:
        logger.info("Using flat mjj distribution for training")
    if cfg.loss=='prob':
        cost_fn=loss.probabilistic_loss#loss.VQC_cost
    else:
        cost_fn=loss.VQC_cost
    VQC = qc.QuantumClassifier(wires=cfg.wires, shots=cfg.shots,dev_name=cfg.device_name,layers=cfg.num_layers)
    VQC.set_circuit()

    NUM_WEIGHTS = len(qc.auto_wires)*3*cfg.num_layers+3 # Extra weight for the bias term in VQC + scale factor for pT
    
    if not cfg.resume:
        init_weights = qc.np.float64(qc.np.random.uniform(0, qc.np.pi, size=(NUM_WEIGHTS,), requires_grad=True))

    train_max_n = cfg.train_n
    valid_max_n = cfg.valid_n
    
    qc.print_training_params()

    # Save initial arguments for logging purposes
    ut.Pickle(cfg, 'args', path=save_dir)
    with open(os.path.join(save_dir, 'args.txt'), 'w+') as f:
        f.write(repr(cfg))

    # Load the data and create a dataloader
    data_key='VQC_train'
    val_key='VQC_val'
    
    logger.info(f'loading data from {ps.PathSetter(data_path=cfg.data_dir).get_data_path(data_key)}')
    train_filelist = sorted(glob.glob(os.path.join(ps.PathSetter(data_path=cfg.data_dir).get_data_path(data_key), '*.h5')))
    val_filelist = sorted(glob.glob(os.path.join(ps.PathSetter(data_path=cfg.data_dir).get_data_path(val_key), '*.h5')))

    if (len(train_filelist) == 0) or (len(val_filelist) == 0):
        raise FileNotFoundError(f"Could not find files in {ps.PathSetter(data_path=cfg.data_dir).get_data_path(data_key)} or {ps.PathSetter(data_path=cfg.data_dir).get_data_path(val_key)}")

    logger.info(f"Training on {len(train_filelist)} files found at {ps.PathSetter(data_path=cfg.data_dir).get_data_path(data_key)}")
    logger.info(f"Validating on {len(val_filelist)} files found at {ps.PathSetter(data_path=cfg.data_dir).get_data_path(val_key)}")

    train_loader = cr.OneP1QDataLoader(filelist=train_filelist, batch_size=cfg.batch_size, input_shape=(len(qc.auto_wires), 3), train=True,
                                            max_samples=train_max_n, normalize_pt=cfg.norm_pt, logger=logger,dataset=cfg.dataset)
    val_loader = cr.OneP1QDataLoader(filelist=val_filelist, batch_size=cfg.batch_size, input_shape=(len(qc.auto_wires), 3), train=False,
                                          max_samples=valid_max_n, normalize_pt=cfg.norm_pt, dataset=cfg.dataset)

    # Initialize the optimizer
    optimizer = qc.qml.AdamOptimizer(stepsize=cfg.lr)

    # Initialize the trainer with WandB
    trainer = qc.QuantumTrainer(VQC, lr=cfg.lr, backend_name=cfg.backend, init_weights=init_weights, device_name=cfg.device_name,
                                train_max_n=train_max_n, valid_max_n=valid_max_n, epochs=cfg.epochs, batch_size=cfg.batch_size,
                                logger=logger, save=cfg.save, patience=cfg.patience, optimizer=optimizer,improv=cfg.improv,\
                                      loss_fn=cost_fn,lr_decay=cfg.lr_decay, wandb=wandb,loss_type=cfg.loss)

    trainer.print_params('Initialized parameters!')
    trainer.set_directories(save_dir)

    if cfg.resume:
        trainer.set_current_epoch(ut.get_current_epoch(model_path))
        logger.info(f"Resuming training from epoch {trainer.current_epoch}")
    else:
        logger.info(f"Training started at {datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}")
        logger.info(f'Epochs: {cfg.epochs} | Learning rate: {cfg.lr} | Batch size: {cfg.batch_size} \nBackend: {cfg.backend} | Wires: {cfg.wires} | Shots: {cfg.shots} \n')    
        logger.info(f'Additional information: {cfg.desc}')

    if cfg.evictable:
        trainer.is_evictable_job(seed=cfg.seed)

    # Begin training
    abs_start = time.time()
    try:
        history = trainer.run_training_loop(train_loader, val_loader)
    except KeyboardInterrupt:
        print("WHYYYYY")
        print("DON'T PRESS CTRL+C AGAIN. I'M TRYING TO SAVE THE CURRENT MODEL AND WRITE TO LOG!")
        trainer.save(save_dir, name='aborted_weights.pickle')
        trainer.print_params('Training aborted. Current parameters are: ')
    finally:
        logger.info('Training completed with the following parameters:')
        trainer.print_params('Trained parameters:')
        history = trainer.fetch_history()
        print(history)
        if cfg.save:
            done_epochs = len(history['train'])
            ut.Pickle(history, 'history', path=save_dir)
            fig, axes = plt.subplots(figsize=(15, 12))
            axes.plot(qc.np.arange(done_epochs), history['train'], label='train', linewidth=2)
            axes.plot(qc.np.arange(done_epochs + 1), history['val'], label='val', linewidth=2)
            axes.set_xlabel('Epochs', size=25)
            axes.set_ylabel('$1-<T|F> $(in %)', size=25)
            axes.set_xticks(qc.np.arange(0, done_epochs + 1, 5))
            axes.legend(prop={'size': 25})

            axes.tick_params(labelsize=20)
            fig.savefig(os.path.join(save_dir, 'history'))
            abs_end = time.time()
            logger.info(f"Training finished at {datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}")
            logger.info(f"Total time taken including all overheads: {abs_end - abs_start:.2f} seconds")

        # Close WandB run
        wandb.finish()

if __name__ == "__main__":
    main()
