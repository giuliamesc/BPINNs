# Bayesian Physics-Informed Neural Networks

## :pushpin: Requirements
- `python` version 3.9.* (download from [here](https://www.python.org/downloads/))
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
3. Enter into the virtual environment with `source my_env_name\bin\activate`:file_folder:
4. Check that the environment is empty with `pip freeze`; normally, it should print nothing
5. Install the required packages from the `.txt` file `requirements.txt` with `pip install -r requirements.txt`:file_folder:
6. Run again `pip freeze` and check that the environment is no longer empty
7. Add the environment folder to your `.gitignore` (in order to avoid pushing the packages on git!)

To exit from the virtual environment, use `deactivate`

## :open_file_folder: Repository Structure

- :file_folder: `config` contains `.json` files which encode the options and parameter choices for the test cases proposed.
- :file_folder: `data` contains, in separate subfolders, the dataset for each test case. In each subfolder you can find `.npy` files storing inputs (`x.npy`), solution (`u.npy`) and parametric field (`f.npy`). The subfolder `trash` contains the dataset generated for the last run that has not been saved.
- :file_folder: `results` contains, in separate subfolders, the results for each test case. In each subfolder, you can find the folders `plot` with the plots and `weights`, with loss history and summary of the experiment options and errors in `.txt` files.
- :file_folder: `src` contains the source code, described in detail in the section below.

## :computer: Source Code 
- :file_folder: `algorithms` is a module containing classes representing the training algorithms proposed in this project: Hamiltionian Monte Carlo (`HMC`), Stein Variational Gradient Descent (`SVGD`) and Variational Inference (`VI`).
- :file_folder: `networks` contains classes for each part of the Bayesian Neural Netwok. The network built is an instance of the class `BayesNN`, which inherits methods and attributes from `LossNN` and `PredNN`, having the loss computation and the prediction/post-processing functionalities, respectively. In turn, the above classes inherit from `CoreNN`, representing a basic fully connected network). The subfolder `equations` contains the differential operators library (`Operators.py`) and, in separate files, the definition of dataset pre and post processing and physical loss for each problem studied.
- :file_folder: `postprocessing` is a module with the classes needed to generate the plots (`Plotter`) and to store results, uncertainty quantification study, loss history and network parameters (`Storage`).
- :file_folder: `setup` is a module containing the class to set parameters (in `param.py`), reading them both from the `.json` files and from command line, which is parsed by the class contained in `args.py`. `data_creation.py` contains the class for dataset creation starting from raw data stored in the folder `data`, and `data_loader` defines the data loader class.
- :file_folder: `utility` contains technical auxiliary tasks, such as the creation of folders (`directiories.py`), the selection of the algorithm (`select_algorithm.py`) and the terminal formatting (`setup.py`).
- `generate_data.py`is the script to be runned before the main for each new test case and it generates a new data subfolder.
- `main.py` is the executable script, relying on all the above.

## :books: References
- *B-PINNs: Bayesian Physics-Informed Neural Networks for Forward and Inverse PDE Problems with Noisy Data*, Liu Yang, Xuhui Meng, George Em Karniadakis, Mar 2020.


## :speech_balloon: Authors 
- Giulia Mescolini ([@giuliamesc](https://gitlab.com/giuliamesc)) 
- Luca Sosta ([@sostaluca](https://gitlab.com/sostaluca))
## :thought_balloon: Tutors
- Stefano Pagani ([@StefanoPagani](https://gitlab.com/StefanoPagani))
- Andrea Manzoni


