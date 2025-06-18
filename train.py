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
import tensorflow as tf
import numpy as np
import tqdm
import quantum.architectures as qc

# Limit TensorFlow's thread usage to be a good citizen on shared servers.
NUM_THREADS = 8
tf.config.threading.set_inter_op_parallelism_threads(NUM_THREADS)
tf.config.threading.set_intra_op_parallelism_threads(NUM_THREADS)

@hydra.main(config_path="./hydra_configs/", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # Set up directories
    base_dir:str=cfg.base_dir
    save_dir = os.path.join(cfg.save_dir, cfg.seed)
    plot_dir = os.path.join(save_dir, 'plots')
    pathlib.Path(plot_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.path.join(save_dir, 'checkpoints_sf')).mkdir(parents=True, exist_ok=True)

    # Initialize WandB
    if cfg.log_wandb:
        try:
            run_str=f"{os.getlogin()}_{cfg.seed}"
        except:
            run_str=f"default_user_{cfg.seed}"
        wandb.init(project="1P1Qm", config=OmegaConf.to_container(cfg), name=run_str,notes=cfg.desc)
        with open(os.path.join(save_dir, "wandb_run_id.txt"), "w") as f:
            f.write(wandb.run.id)

    # Logging setup
    logger.add(os.path.join(save_dir, 'logs.log'), rotation='10 MB', backtrace=True, diagnose=True, level='DEBUG', mode="w")
    logger.info("########################################### \n\n")
    
    print("Will save models to: ", save_dir)

    # Save initial arguments for logging purposes
    ut.Pickle(cfg, 'args', path=save_dir)
    with open(os.path.join(save_dir, 'args.txt'), 'w+') as f:
        f.write(OmegaConf.to_yaml(cfg))

    # --- Corrected Initialization ---
    logger.info("Initializing circuit parameters...")
    # The circuit template uses 4 params per wire (disp_mag, disp_phase, squeeze_mag, squeeze_phase)
    # Override the config value if it's incorrect.
    if cfg.params_per_wire != 4:
        logger.warning(f"Config `params_per_wire` is {cfg.params_per_wire}, but circuit requires 4. Overriding to 4.")
        cfg.params_per_wire = 4
    qc.initialize(wires=cfg.wires, layers=cfg.num_layers, params=cfg.params_per_wire)



    if not cfg.resume:
        logger.info("Starting new training run.")
        # Save a snapshot of the code for reproducibility
        tmpfile = os.path.join(save_dir, 'FROZEN_ARCHITECTURE.py')
        subprocess.run(['cp', os.path.join(base_dir,'quantum/architectures.py') ,tmpfile])
        tmpfile = os.path.join(save_dir, 'FROZEN_DATAREADER.py')
        subprocess.run(['cp', os.path.join(base_dir,'case_reader.py'), tmpfile])
        tmpfile = os.path.join(save_dir, 'FROZEN_LOSS.py')
        subprocess.run(['cp', os.path.join(base_dir,'quantum/losses.py'), tmpfile])

    # --- Instantiate Model ---
    if cfg.cutoff_dimension is None:
        raise ValueError("Configuration 'cutoff_dimension' cannot be null; it is required for the Fock backend.")
    
    model = qc.QuantumClassifier(wires=cfg.wires, cutoff_dim=cfg.cutoff_dimension)

    # --- Weights Initialization ---
    # The circuit template uses weights[-3] for scaling, so we need at least 3 extra weights.
    if cfg.extra_weights < 3:
        logger.warning(f"Config `extra_weights` is {cfg.extra_weights}, but circuit requires at least 3. Setting to 3.")
        cfg.extra_weights = 3
    
    NUM_WEIGHTS = len(qc.auto_wires) * cfg.params_per_wire * cfg.num_layers + cfg.extra_weights
    logger.info(f"Circuit requires {NUM_WEIGHTS} weights.")
    init_weights = np.random.uniform(0, np.pi, size=(NUM_WEIGHTS,)).astype(np.float32)


    # --- Optimizer and Loss ---
    optimizer = tf.keras.optimizers.Adam(learning_rate=cfg.lr)
    if cfg.loss == 'MSE':
        loss_fn = loss.mean_squared_error
    elif cfg.loss == 'BCE':
        loss_fn = loss.binary_crossentropy
    else:
        raise ValueError(f"Unsupported loss type: {cfg.loss}")

    # --- Trainer ---
    logger.info("Instantiating QuantumTrainer...")
    trainer = qc.QuantumTrainer(
        model=model,
        optimizer_tf=optimizer,
        loss_fn_tf=loss_fn,
        init_weights_val=init_weights,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        logger=logger,
        save=cfg.save,
        patience=cfg.patience,
        improv=cfg.improv,
        wandb_run=wandb.run if cfg.log_wandb else None,
        save_dir_path=save_dir
    )

    if cfg.resume:
        # NOTE: Resuming logic for this custom trainer needs to be implemented.
        # For now, it will just start a new training run with the loaded config.
        pass


    # --- Data Loading ---
    data_key='VPC_train'
    val_key='VPC_val'
    
    logger.info(f'loading data from {ps.PathSetter(data_path=cfg.data_dir).get_data_path(data_key)}')
    train_filelist = sorted(glob.glob(os.path.join(ps.PathSetter(data_path=cfg.data_dir).get_data_path(data_key), '*.h5')))
    val_filelist = sorted(glob.glob(os.path.join(ps.PathSetter(data_path=cfg.data_dir).get_data_path(val_key), '*.h5')))

    if (len(train_filelist) == 0) or (len(val_filelist) == 0):
        raise FileNotFoundError(f"Could not find files in the specified data directories.")

    logger.info(f"Training on {len(train_filelist)} files, validating on {len(val_filelist)} files.")

    train_loader = cr.OneP1QDataLoader(filelist=train_filelist, batch_size=cfg.batch_size, input_shape=(cfg.wires, 3), train=True,
                                            max_samples=cfg.train_n, normalize_pt=cfg.norm_pt, logger=logger,dataset=cfg.dataset)
    val_loader = cr.OneP1QDataLoader(filelist=val_filelist, batch_size=cfg.batch_size, input_shape=(cfg.wires, 3), train=False,
                                          max_samples=cfg.valid_n, normalize_pt=cfg.norm_pt, dataset=cfg.dataset)



    # --- Pre-run Confirmation ---
    logger.info("--- Training Configuration ---")
    logger.info(f"Epochs: {cfg.epochs}, Batch Size: {cfg.batch_size}, Learning Rate: {cfg.lr}")
    logger.info(f"Wires: {cfg.wires}, Layers: {cfg.num_layers}, Params per Wire: {cfg.params_per_wire}, (Cutoff): {cfg.cutoff_dimension}")
    logger.info(f"Loss Function: {cfg.loss}, Using Flat Data: {cfg.flat}")
    logger.info("-----------------------------")
    confirm = input("Press Enter to begin training or 'q' to quit: ")
    if confirm.lower() == 'q':
        logger.warning("Training cancelled by user.")
        return

    # --- Training ---
    trainer.run_training_loop(
        train_loader=train_loader,
        val_loader=val_loader
    )

    logger.info("Training finished.")
    if cfg.log_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()
