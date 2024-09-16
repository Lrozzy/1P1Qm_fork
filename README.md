# Anomaly detection in high-energy physics using a quantum autoencoder

### Authors: Aritra Bal (KIT Karlsruhe) and Benedikt Maier (Imperial College, London)

Derived using the methodology of [arxiv: 2112.04958](https://arxiv.org/abs/2112.04958).

To train the autoencoder: 

    python3 train.py --train --wires 8 --trash-qubits 5 -b 10 -e 15 --backend autograd --save --seed $RANDOM --lr 0.005 --desc "ENTER YOUR DESCRIPTION HERE" --train_n 5000 --valid_n 1000


It is possible to speed up training by using multiple cores. To achieve this, set the device by passing `--device_name lightning.kokkos --num_threads N` in the arguments, where `N` is the number of threads you wish to use.

Notes: 
- The argument `--seed` (set here to `$RANDOM`) is used to identify a given training run, which is then further described by the text following `--desc`
- The directories where the input files are stored, and where the results are stored, can be set by modifying `helpers.utils.path_dict` 
- If `seed = S`, then a new subdirectory is created in the base save directory at the path `/path/to/base/directory/S` and your results are saved there.   
- The data loader is defined in `case_reader.py`, the quantum circuit architecture is defined in `quantum.architecture` and the loss function is defined in `quantum.losses`. Feel free to modify/add to it!

