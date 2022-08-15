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
- :file_folder: `src` contains the source code, described in the section below.

## :computer: Source Code 

## :books: References

## :speech_balloon: Authors 
- Giulia Mescolini ([@giuliamesc](https://gitlab.com/giuliamesc)) 
- Luca Sosta ([@sostaluca](https://gitlab.com/sostaluca))
## :thought_balloon: Tutors
- Stefano Pagani ([@StefanoPagani](https://gitlab.com/StefanoPagani))
- Andrea Manzoni


