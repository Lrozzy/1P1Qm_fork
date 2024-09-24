# Anomaly detection in high-energy physics using a quantum autoencoder

### Authors: Aritra Bal (KIT Karlsruhe) and Benedikt Maier (Imperial College, London)

Derived using the methodology of [arxiv: 2112.04958](https://arxiv.org/abs/2112.04958).

To train the autoencoder: 

    python3 case_qml.py --train --wires 10 \
    --trash-qubits 7 -b 250 -e 15 --backend autograd --save --seed ${seed} --lr 0.005 --desc "Using arbitrary 3D rotations and 3x weights" \
    --train_n 75000 --valid_n 15000

The example above can be run on multiple cores. Set the device to `kokkos` by passing `--device_name lightning.kokkos` in the arguments. In addition, set the values of the `OMP_PROC_BIND` and `OMP_NUM_THREADS` environment variables **before** you run the python script.

    export OMP_PROC_BIND=true
    export OMP_NUM_THREADS= <N_THREADS>

In addition, don't forget to set the `PYTHONPATH` environment variable such that it includes the base directory `qae_hep` (or whatever you named it).

It is possible to use a GPU for accelerated training as well. To do this, change the argument to `--device lightning.gpu`. Note that for GPU acceleration, a GPU with Compute Capability $>=7.0$ and CUDA Version $>= 12.0$ is needed. 

You can use the docker container here:

    docker pull neutrinoman4/qml-lightning.gpu:v2.0


Notes: 
- The argument `--seed` (set here to `$RANDOM`) is used to identify a given training run, which is then further described by the text contained in the `--desc` argument.
- The directories where the input files are stored, and where the results are stored, can be set by modifying `helpers.utils.path_dict` 
- If `seed = S`, then a new subdirectory is created in the base save directory at the path `/path/to/base/directory/S` and your results are saved there.   
- The data loader is defined in `case_reader.py`, the quantum circuit architecture is defined in `quantum.architecture` and the loss function is defined in `quantum.losses`. Feel free to modify/add to it!

