# Anomaly detection in high-energy physics using a quantum autoencoder

Authors: Aritra Bal (KIT Karlsruhe) and Benedikt Maier (Imperial College, London)

------

## Multi-core Training

In this repository, we use Hydra for managing parameters, and WandB for experiment tracking.  

Assuming that your config YAML file is placed in the directory `$HYDRA_CONF`, and has the name `config.yaml`, you can start training by running:
    
    python3 train.py --config-path $HYDRA_CONF --config-name config

The example above can be run on multiple cores by setting the `lightning.kokkos` device in your configuration YAML. In addition, set the values of the `OMP_PROC_BIND` and `OMP_NUM_THREADS` environment variables **before** you run the python script.

    export OMP_PROC_BIND=spread
    export OMP_NUM_THREADS= <N_THREADS>

Examples of config YAML files can be found in the directory `hydra_configs`.
In addition, don't forget to set the `PYTHONPATH` environment variable such that it includes the base directory `qae_hep` (or whatever you named it).

## Recommended: GPU Training

It is possible to use a GPU for accelerated training as well. To do this, you must override the corresponding argument `device=lightning.gpu`. Note that for GPU acceleration with Pennylane, a GPU with Compute Capability $>=7.0$ and CUDA Version $>= 12.0$ is needed.

You can use the docker container here:

    docker pull neutrinoman4/qml-lightning.gpu:v5.0

Note: Check the version, it MUST be v5.0

For running on Horeka@KIT (Slurm-based), read the documentation [here](https://www.nhr.kit.edu/userdocs/ftp/containers/).

To run GPU jobs on an HTCondor cluster, take a look at the scripts in `condor_example/`.

If your HPC cluster supports Singularity/Apptainer, then you may use it as follows:

     apptainer build qml-lightning-gpu.sif docker://neutrinoman4/qml-lightning.gpu:v4.0
     apptainer shell qml-lightning-gpu.sif

The Docker image is derived from the Pennylane Lightning GPU v0.38.0 docker image, and contains some additional relevant libraries.

After you're all set up, run your code using the `train.py` script as shown below. The quantum circuit architecture is loaded from `quantum/architectures.py` and the loss function from `quantum/losses.py`. At the moment, two circuit architectures are defined: `circuit()` and `reuploading_circuit()`. You are free to try out new circuit architectures. The `train.py` method copies the current version of the `architecture.py` file to the `save_dir` as `save_dir/FROZEN_ARCHITECTURE.py`, in case you edit this file in between runs.

    python3 train.py --config-path $HYDRA_CONF --config-name config device=lightning.gpu

## Inference

This can be run on either CPU or GPU, though using GPU does not lead to any noticeable improvements. The same hydra config that was used for training the network can be reused during inference, with some additional arguments as follows:

    python3 test.py --config-path $HYDRA_CONF --config-name config +signal=grav_1p5_narrow

There are more arguments that you can see in the examples, such as `log_wandb`, which logs all images to the same WandB run used to track the training.

## Notes

- The argument `seed` is used to identify a given training run, which is then further described by the text contained in the `desc` argument.
- If `seed = S`, then a new subdirectory is created in the base save directory at the path `/path/to/base/directory/S` and your results are saved there.
- The data loader is defined in `case_reader.py`, the quantum circuit architecture is defined in `quantum.architecture` and the loss function is defined in `quantum.losses`. Feel free to modify/add to it!

You can download (a subset of) the training data [here](https://drive.google.com/drive/folders/1fGATNxxcCKPk6mZ54Ucv1mYZteOnh33-?usp=sharing).  

### Citations

1. [arxiv: 2112.04958](https://arxiv.org/abs/2112.04958)
