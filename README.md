# Pressure-based Soft Agents
This is the official repository for the ALife'22 (Artificial Life conference, 2022) paper "Shape Change and Control of Pressure-based Soft Agents", hosting all the code for replication. This work was carried out at the Evolutionary Robotics and Artificial Life lab (ERRALab) at University of Trieste, Italy.

## Installation
Clone the repo:
```
git clone https://github.com/pigozzif/SoftBodies.git
```
### Requirements
Either install Python dependencies with conda:
```
conda env create -f environment.yml
conda activate pybox2d
```
or with pip:
```
pip install -r requirements.txt
```

## Scope
By running
```
python main.py
```
you will launch an evolutionary optimization for the controller (an artificial neural network) of Pressure-based Soft Agents (PSAs): they are bodies of gas enveloped by a chain of springs and masses, with pressure pushing on the masses from inside the body. Pressure endows the agents with structure, while springs and masses simulate softness and allow the agents to assume an infinite gamut of shapes. Actuation takes place by changing the length of springs or modulating global pressure. 
At the same time, evolution metadata will be saved inside the `output` folder.

## Usage
## Usage
Inside `config.yaml` you may edit the following parameters:
Argument         | Type                                | Default
-----------------|-------------------------------------|-------------------------
n_masses         | integer                             | 20
r                | float                               | 10
size             | string                              | large
solver           | {cmaes,ga,es}                       | cmaes
task             | {flat,hilly-1-10,escape}            | escape
evaluations      | integer                             | 10000
mode             | {random,opt-parallel,best,inflate}  | random
seed             | integer                             | 0
np               | integer                             | 1
control_pressure | {0,1}                               | 1
save_video       | {0,1}                               | 0

where {...} denotes a finite and discrete set of possible choices for the corresponding argument. The description for each argument is as follows:
* n_masses: the number of rigid masses in the envelope.
* r: the radius of the agent.
* size: label for the size of the agent (just for naming the logs dir).
* solver: the evolutionary algorithm to perform optimization with.
* task: the task to experiment with.
* evaluations: the total number of fitness evaluations before stopping evolution.
* mode: `random` stands for a random controller, `best` loads the `.npy` file for the corresponding experiment, `opt-parallel` is full-fledged evolution from scratch.
* seed: the random seed.
* np: the number of processes to perform evolution with. Parallelization is taken care by the code and implements a distributed fitness assessment.
* control_pressure: if 1, control also pressure, otherwise just the springs length.
* save_video: if 1, saves simulation to `video.mp4`.

## Bibliography
Please cite as:
```
@article{pigozzi2022shape,
  title = {Shape Change and Control of Pressure-based Soft Agents},
  author = {Pigozzi, Federico},
  journal={arXiv preprint arXiv:2205.00467},
  year = {2022}
}
```
