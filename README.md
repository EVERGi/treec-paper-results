# TreeC paper results

This repository contains the code to reproduce and visualise the results of the paper [TreeC: a method to generate interpretable energy management systems using a metaheuristic algorithm](https://arxiv.org/abs/2304.08310). The tree models obtained for the paper are available in the anm6easy_paper_trees and boptest_paper_trees folders. These models can be evaluated and visualised using the command line tool (treec_results.py) and new model can be generated  using the same tool.

## Installation

We recommend using [anaconda](https://www.anaconda.com/), create a new python environment and running the different commands below through the anaconda prompt.

To be able to run the results, you should first get the git submodules for the BOPTEST simulator with:

```
git submodule update --init --recursive
```

Then install the dependencies by running:

```
pip install -r requirements.txt
```

To execute the boptest simulator, you need to install make and docker.

For Linux install these via your package managers.

For Windows we recommend installing make through [Chocolatey](https://chocolatey.org/).
To install Chocolatey, you can run the following command in a powershell opened as administrator:

```
Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
```
Once Chocolatey is installed you can run the following command in the powershell to install make:
```
choco install make
```
For Docker install it through the official executable on their website [https://www.docker.com/](https://www.docker.com/) and restart your computer:

In case it did not work, refer to the boptest v0.1.0 github repository page for further indication [https://github.com/ibpsa/project1-boptest/tree/v0.1.0?tab=readme-ov-file](https://github.com/ibpsa/project1-boptest/tree/v0.1.0?tab=readme-ov-file) 

Once make and docker are installed you can build and run the boptest simulator with:
```
cd boptest/project1-boptest
make build TESTCASE=bestest_hydronic_heat_pump PORT=5000
make run TESTCASE=bestest_hydronic_heat_pump PORT=5000
```

This installation guide has been tested successfully on Windows with conda version 4.9.2, python version 3.11.9, make version 4.4.1 and docker version 27.3.1.

## Training reproduction and visualisation

Use the treec_results.py command line tool to reproduce the results, a description of all the options of the tool is available in the --help menu of the tool.

Here below are examples of commands to reproduce paper visualisations and trainings.

Visualise simulation of the best performing tree displayed in the paper's Figure 4 for ANM6easy case:
```
python treec_results.py -c anm -m visu -s 0 -p anm6easy_paper_trees/ANM6Easy-v0_tree_39/tree_model.txt
```
Visualise simulation of tree displayed in the paper's Figure 6 for BOPTEST case:
```
python treec_results.py -c bop -m visu -s 0 -p boptest_paper_trees/case_E_tree_10/tree_model.txt
```
Run one training for ANM6easy case with seed 100:
```
python treec_results.py -c anm -m train -s 100 -t 1500
```
Run one training for BOPTEST case for peak_heat_day and constant price (scenario 0):
```
python treec_results.py -c bop -m train -s 0 -t 150
```

## Reproduce results
Many of these instructions can be done in parallel. How to do this in parallel is often specific to the computer on which the training is ran.
### ANM6Easy case
Run the following command 5 times:
```
python treec_results.py -c anm -m train -s 100 -t 1500
```
Then select the tree model with the lowest score and you will get the EMS for seed 100 (value after ``-s`` in command). Do the same process for seeds 100 to 119 to obtain 20 different EMSs as was done in the original paper.

Now evaluate these EMSs using the validation score printed through the following command (replace ``PATH_TO_BEST_TREE_MODEL`` with the path to tree model of the generated EMS):
```
python treec_results.py -c anm -m visu -s 0 -p PATH_TO_BEST_TREE_MODEL
```
Execute this command for seeds from 0 to 10 and average the printed validation scores. This will give you a comparable score to the ones presented in the paper.
### Boptest case
Run the following command 5 times:
```
python treec_results.py -c bop -m train -s 0 -t 150
```
Then select the tree model with the lowest score and you will get the EMS for scenario 0 (value after ``-s`` in command). Scenario 0 corresponds to the peak_heat_day time period with constant electricity price. In the original paper two EMSs were generated for each of the 6 scenarios.

Now evaluate these EMSs using the validation scores printed through the following command (replace ``SCENARIO_MODEL`` with the scenario number the EMS was trained on and ``PATH_TO_BEST_TREE_MODEL`` with the path to tree model of the generated EMS).
```
python treec_results.py -c bop -m visu -s SCENARIO_MODEL -p PATH_TO_BEST_TREE_MODEL
```
This will give you the total discomfort and total electricity cost comparable to the ones presented in the paper.
