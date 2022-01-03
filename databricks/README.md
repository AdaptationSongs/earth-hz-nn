# Databricks notebooks

## Installation 
1. Import .py files into your Databricks workspace
2. Create a job to run run-train-model on the schedule you choose
3. Set the following enviromental variables on your cluster:
### Environment variables
DB_HOST=
DB_NAME=
DB_USER=
DB_PASSWORD=
ADLS_ACCOUNT=
ADLS_CLIENT_ID=
ADLS_TENANT=
ADLS_CLIENT_SECRET=
CLIP_SECS=
4. Install the following additonal libraries on your cluster:
### Libraries
azure-datalake-store
imageio
scikit-image
scipy
soundfile
sqlalchemy
h5py==2.10.0
Keras==2.3.0
librosa==0.6.3
numba==0.48.0
pandas>=0.25
tensorflow==1.14.0

panotti from https://github.com/drscotthawley/panotti

## Cluster configuration
Tested on Databricks 7.3 LTS, worker type Standard_DS3_v2
