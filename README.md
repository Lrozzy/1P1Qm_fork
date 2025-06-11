# 1P1Qm: One Particle - One Qubit but on Qumodes 

#### Aritra Bal<sup>1</sup>, Benedikt Maier<sup>2</sup>
1. Karlsruhe Institute of Technology, KIT, DE
2. Imperial College, London, UK

------

## Prerequisites:
 We recommend the usage of miniconda for managing python environments. Install miniconda following the instructions [here](https://www.anaconda.com/docs/getting-started/miniconda/install#linux).

Thereafter, run the following commands:

    conda env create -f quantum-photonic.yaml
    conda activate quantum-photonics

The required list of packages can be seen in the `requirements.txt` file, which is called upon by conda. 

## Training on CPU

In this repository, we use Hydra for managing parameters, and WandB for experiment tracking.  

Assuming that your config YAML file is placed in the directory `$HYDRA_CONF`, and has the name `config.yaml`, you can start training by running:
    
    python3 train.py --config-path $HYDRA_CONF --config-name config


Examples of config YAML files can be found in the directory `hydra_configs`.
In addition, don't forget to set the `PYTHONPATH` environment variable such that it includes the base directory (this can be whatever you named it).

## Inference

The same hydra config file that was used for training the network can be reused during inference. You may need to specify how many events to train on. 

    python3 test_jetclass.py --config-path $HYDRA_CONF --config-name config.yaml read_n=N

There are more arguments that you can see in the examples, such as `log_wandb`, which logs all images to the same WandB run used to track the training.

## Notes

- The argument `seed` is used to identify a given training run, which is then further described by the text contained in the `desc` argument.
- If `seed = S`, then a new subdirectory is created in the base save directory at the path `/path/to/base/directory/S` and your results are saved there.
- The data loader is defined in `case_reader.py`, the quantum circuit architecture is defined in `quantum.architectures` and the loss function is defined in `quantum.losses`. Feel free to modify/add to it!
