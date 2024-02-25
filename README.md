# Pressure-based Soft Agents
This is the official repository for the *ALIFE'22* (Artificial Life conference, 2022) paper and its *Artificial Life* journal extension

**<a href="https://direct.mit.edu/artl/article-abstract/doi/10.1162/artl_a_00415/118225/Pressure-Based-Soft-Agents?redirectedFrom=fulltext">Pressure-based Soft Agents</a>**
<br>
<a href="https://pigozzif.github.io">Federico Pigozzi</a>
<br>

**<a href="https://arxiv.org/abs/2205.00467">Shape Change and Control of Pressure-based Soft Agents</a>**
<br>
<a href="https://pigozzif.github.io">Federico Pigozzi</a>
<br>

hosting all the code for replication. More videos available at this [link](https://pressuresoftagents.github.io).

<div align="center">
<img src="teaser.gif"></img>
</div>

## Installation
Clone the repo:
```
git clone https://github.com/pigozzif/PressureSoftAgents.git
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
Inside `config.yaml` you may edit the following parameters:
Argument         | Type                                | Default
-----------------|-------------------------------------|-------------------------
n_masses         | integer                             | 20
r                | float                               | 10
size             | string                              | large
solver           | {cmaes,ga,es}                       | cmaes
task             | {flat,hilly-1-10,escape,carrier}    | escape
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
@article{pigozzi2023pressure,
  title={Pressure-based soft agents},
  author={Pigozzi, Federico},
  journal={Artificial life},
  pages={1--19},
  year={2023},
  publisher={MIT Press One Broadway, 12th Floor, Cambridge, Massachusetts 02142, USA~…}
}
```
```
@proceedings{pigozzi2022shape
    author = {Pigozzi, Federico},
    title = "{Shape Change and Control of Pressure-based Soft Agents}",
    volume = {ALIFE 2022: The 2022 Conference on Artificial Life},
    year = {2022},
    doi = {10.1162/isal_a_00520}
}
```
