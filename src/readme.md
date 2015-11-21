## Dependency
OS : Amazon Linux AMI release 2015.03  

Python 2.7.9  
Vowpal Wabbit 8.0.0  
Xgboost v0.40  

used python packages are below:  
ipython 3.2.1  
pandas 0.16.2  
numpy 1.9.2  
scipy 0.16.2  
scikit-learn 0.16.1  

## Data
Add Data from Kaggle into "input" folder.  
(except prefecture_locations.csv. BOM-removed file is already there.)

## Run
Run by "ipython batch.py" in the src folder.  
It takes around 2 hours on AWS r3.2xlarge. (Requires 64GB RAM)
