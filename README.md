# Garage demand forecasts
A notebook on modeling a time series for predicting the demand of parking lots of italian garage in the seaside.

## How to run the notebook locally
### Create and activate a new virtual python enviroment
#### Using uv
1. Execute `uv sync` in the project directory root.
it Create a python virtual env in the `.venv`cfolder at the root of the project directory
2. Activate the python virtual enviroment `source .venv/bin/activate`
3. Execute the notebook with your ide.
#### Using bare python
1. Ensure that your version matches that of specified by the pyproject.toml file.
2. Create a python venv `python -m venv .venv`
3. Activate it `source .venv/bin/activate`
4. Install packages `pip install -r requirements.txt`
### Run the notebook
#### For vscode users
1. Run `code timeseries_modeling.ipynb` 
2. Select the (recommended) .venv kernel
#### For jupyter lab users
1. Install the ipykernel package -- it is a dev dependency
2. Activate the project venv
3. Add the venv to the ipython kernel `python -m ipykernel install --user --name=garage-demand-forecasts-venv`
4. Run `jupyter lab` 
5. Select the just installed ipykernel.