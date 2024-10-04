# Anomaly detection in high-energy physics using a quantum autoencoder

Authors: Aritra Bal (KIT Karlsruhe) and Benedikt Maier (Imperial College, London)

------

## Multi-core Training

Not recommended (see GPU usage below)
To train the autoencoder:

    python3 case_qml.py --train --wires 10 \
    --trash-qubits 7 -b 250 -e 15 --backend autograd --save --seed ${seed} --lr 0.005 --desc "Using arbitrary 3D rotations and 3x weights" \
    --train_n 75000 --valid_n 15000

The example above can be run on multiple cores. Set the device to `kokkos` by passing `--device_name lightning.kokkos` in the arguments. In addition, set the values of the `OMP_PROC_BIND` and `OMP_NUM_THREADS` environment variables **before** you run the python script.

    export OMP_PROC_BIND=true
    export OMP_NUM_THREADS= <N_THREADS>

In addition, don't forget to set the `PYTHONPATH` environment variable such that it includes the base directory `qae_hep` (or whatever you named it).

## Recommended: GPU Training

It is possible to use a GPU for accelerated training as well. To do this, change the argument to `--device lightning.gpu`. Note that for GPU acceleration with Pennylane, a GPU with Compute Capability $>=7.0$ and CUDA Version $>= 12.0$ is needed.

You can use the docker container here:

    docker pull neutrinoman4/qml-lightning.gpu:v3.0

For running on Horeka@KIT (Slurm-based), read the documentation [here](https://www.nhr.kit.edu/userdocs/ftp/containers/).
To run GPU jobs on an HTCondor cluster, take a look at the scripts in `condor_example/`.

The Docker image is derived from the Pennylane Lightning GPU v0.38.0 docker image, and contains additional relevant libraries.
To run your code, use the `train.py` script as shown below. While this script does almost the same thing as `case_qml.py` does, it is not self-contained. The quantum circuit architecture is loaded from `quantum/architectures.py` and the loss function from `quantum/losses.py`. At the moment, two circuit architectures are defined: `circuit()` and `reuploading_circuit()`. You are free to try out new circuit architectures. The `train.py` method copies the current version of the `architecture.py` file to the `save_dir` as `save_dir/FROZEN_ARCHITECTURE.py`, in case you edit this file in between runs.

    python3 train.py --train --wires ${QUBITS} --trash-qubits ${TRASH} -b 100 -e 15 --backend "autograd" --save --seed ${seed} --lr 0.005 --desc "'${DESC}'" --train_n ${TRAIN_N} --valid_n ${VALID_N} --save_dir ${SAVE_DIR} --data_dir ${DATA_DIR} --device lightning.gpu

To get a description of the possible options, run `python3 train.py --help`

## Notes

- The argument `--seed` (set here to `$RANDOM`) is used to identify a given training run, which is then further described by the text contained in the `--desc` argument.
- If `seed = S`, then a new subdirectory is created in the base save directory at the path `/path/to/base/directory/S` and your results are saved there.
- The data loader is defined in `case_reader.py`, the quantum circuit architecture is defined in `quantum.architecture` and the loss function is defined in `quantum.losses`. Feel free to modify/add to it!

You can download (a subset of) the training data [here](https://drive.google.com/drive/folders/1fGATNxxcCKPk6mZ54Ucv1mYZteOnh33-?usp=sharing).  

### Citations

1. [arxiv: 2112.04958](https://arxiv.org/abs/2112.04958)
