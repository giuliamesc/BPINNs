# Bayesian Physics-Informed Neural Networks

## :pushpin: Requirements
- `python` version 3.10.* (download from [here](https://www.python.org/downloads/))
- `virtualenv` version 20.14.* (download from [here](https://virtualenv.pypa.io/en/latest/installation.html#via-pip))

## :gear: Installation 
### Windows
1. Go into the directory of your project with `cd project_folder_path`
2. Create an empty *virtual environment* with `py -m venv .\my_env_name`
3. Enter into the virtual environment with `my_env_name\scripts\activate`
4. Check that the environment is empty with `pip freeze`; normally, it should print nothing
5. Install the required packages from the `.txt` file `requirements.txt` with `pip install -r requirements.txt`
6. Run again `pip freeze` and check that the environment is no longer empty
7. Add the environment folder to your `.gitignore` (in order to avoid pushing the packages on git!)

To exit from the virtual environment, use `deactivate`
### Mac and Linux
1. Go into the directory of your project with `cd project_folder_path`
2. Create an empty *virtual environment* with `virtualenv .\my_env_name`
3. Enter into the virtual environment with `source my_env_name\bin\activate`
4. Check that the environment is empty with `pip freeze`; normally, it should print nothing
5. Install the required packages from the `.txt` file `requirements.txt` with `pip install -r requirements.txt`
6. Run again `pip freeze` and check that the environment is no longer empty
7. Add the environment folder to your `.gitignore` (in order to avoid pushing the packages on git!)

To exit from the virtual environment, use `deactivate`

## :open_file_folder: Repository Structure

- :file_folder: `config` contains `.json` files which encode the options and parameter choices for the test cases proposed.
- :file_folder: `data` contains the dataset for each test case. In each subfolder, you can find `.npy` files storing inputs, solution and parametric field values. They are stored separately, according to the category of data they belong to (such as fitting data, collocation data...). <br /> 
In particular, we have:
    - *Boundary data*: `dom_bnd.npy`, `sol_bnd.npy` for the coordinates and solution values of boundaries;
    - *Collocation data*: `dom_pde.npy` for the coordinates of collocation points;
    - *Fitting data*: `dom_sol.npy`, `sol_train.npy` and `dom_par.npy`, `par_train.npy` for coordinate and values couples for solution and/or parametric field;
    - *Test data*: `dom_test.npy`, `sol_test.npy`, `par_test.npy` for coordinate and function values of test points.
- :file_folder: `outs` contains the results for each test case. In each subfolder, you can find the folders:
    - `log` with loss history and summary of the experiment options and errors in `.txt` files 
    - `plot` with the plots
    - `thetas` with network parameters
    - `values` with the solution computed by the network
- :file_folder: `src` contains the source code, described in detail in the section below.

## :computer: Source Code 
- `main.py` is the executable script, relying on all the below modules.
- `main_data.py` is a script that can be run independently from the main and it generates a new data subfolder for each new test case.
- `main_loader.py` is a script that can be run independently from the main and it reloads a test case already present in `outs`.
- :file_folder: `setup` is a module containing:
    - the class to parse command line arguments (in `args.py`)
    - the class to set parameters (in `param.py`), reading them both from the configuration files and from command line
    - `data_creation.py`  contains the class for dataset creation starting from raw data stored in the folder `data`
    - `data_generator.py` contains the class to generate raw data 
    - `data_loader.py` defines the data loader class (now not in use)
- :file_folder: `utility` contains technical auxiliary tasks, in particular: 
    - `switcher.py` switches to the test case under analysis among the files contained in the three folders below
    - `directories.py` generates directories for data generation and results storage
    - `miscellaneous.py` contains utility functions for all other purposes 
- :file_folder: `algorithms` is a module containing classes representing the training algorithms proposed in this project:
    - `ADAM`: Adaptive Moment Estimation 
    - `HMC`: Hamiltionian Monte Carlo
    - `SVGD` (WIP): Stein Variational Gradient Descent
    - `VI` (WIP): Variational Inference <br />
    The folder includes as well the class `Trainer`, used to manage the pre-training and training algorithm.
- :file_folder: `datasets` contains: 
    - :file_folder: `config` contains definitions of functions and domains for generating datasets
    - :file_folder: `template` contains names and definitions of input and output for generating datasets
- :file_folder: `equations` contains the differential operators library (`Operators.py`) and, in separate files, the definition of physical loss and dataset pre/post processing criteria for each problem studied
- :file_folder: `networks` contains classes for each part of the Bayesian Neural Netwok. 
    The network built is an instance of the class `BayesNN`, which inherits methods and attributes from `LossNN` and `PredNN`, having the loss computation and the prediction/post-processing functionalities, respectively. In turn, the above classes inherit from `CoreNN`, representing a basic fully connected network)
- :file_folder: `postprocessing` is a module with:
    - the class `Plotter` to generate the plots and save them in the folder `outs`
    - the class `Storage` to store and load results, uncertainty quantification study, loss history and network parameters 

## :books: References
- *B-PINNs: Bayesian Physics-Informed Neural Networks for Forward and Inverse PDE Problems with Noisy Data*, Liu Yang, Xuhui Meng, George Em Karniadakis, Mar 2020.
- *Bayesian Physics-Informed Neural Networks for Inverse Uncertainty Quantification problems in Cardiac Electrophysiology*, Master Thesis at Politecnico di Milano by Daniele Ceccarelli.
- *A physics-informed deep learning framework for inversion and surrogate modeling in solid mechanics*, Ehsan Haghighat, Maziar Raissi, Adrian Moure, Hector Gomez, Ruben Juanes, Mar 2021.

## :speech_balloon: Authors 
- Giulia Mescolini ([@giuliamesc](https://gitlab.com/giuliamesc)) 
- Luca Sosta ([@sostaluca](https://gitlab.com/sostaluca))
## :thought_balloon: Tutors
- Stefano Pagani ([@StefanoPagani](https://gitlab.com/StefanoPagani))
- Andrea Manzoni


