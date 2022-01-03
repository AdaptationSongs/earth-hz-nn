# Databricks notebook source
dbutils.widgets.text('iteration', '', 'Iteration')
iteration = int(dbutils.widgets.get('iteration'))

# COMMAND ----------

import os

class Config(object):
    CLIP_SECS = float(os.environ.get('CLIP_SECS') or 4)

    # Database
    DB_HOST = os.environ.get('DB_HOST')
    DB_NAME = os.environ.get('DB_NAME')
    DB_USER = os.environ.get('DB_USER')
    DB_PASSWORD = os.environ.get('DB_PASSWORD')

    # Azure Data Lake
    ADLS_ACCOUNT = os.environ.get('ADLS_ACCOUNT')
    ADLS_TENANT = os.environ.get('ADLS_TENANT')
    ADLS_CLIENT_ID = os.environ.get('ADLS_CLIENT_ID')
    ADLS_CLIENT_SECRET = os.environ.get('ADLS_CLIENT_SECRET')

# COMMAND ----------

import sys
import pandas as pd
import psycopg2 as pg
import soundfile as sf
from pathlib import Path
## Required for Azure Data Lake Storage Gen1 filesystem management
from azure.datalake.store import core, lib, multithread

# Load configuration from .env file
conf = Config()

# Azure Datalake authentication
adlCreds = lib.auth(tenant_id=conf.ADLS_TENANT, client_secret=conf.ADLS_CLIENT_SECRET, client_id=conf.ADLS_CLIENT_ID, resource='https://datalake.azure.net/')
# Create a filesystem client object
adl = core.AzureDLFileSystem(adlCreds, store_name=conf.ADLS_ACCOUNT)

# Connect to database
conn_string = 'host={host} dbname={name} user={user} password={password}'.format(host=conf.DB_HOST, name=conf.DB_NAME, user=conf.DB_USER, password=conf.DB_PASSWORD)
conn = pg.connect(conn_string)

# COMMAND ----------

# Set some enviromental variables for the shell
os.environ['ITER'] = str(iteration)
os.environ['DUR'] = str(conf.CLIP_SECS)

prefix = 'iterations/' + str(iteration)

# COMMAND ----------

# Outputs standard sized .wav files for a row from input dataframe
def gen_clip(row):
    fpath = row['path'] + '/' + row['file_name']
    with adl.open(fpath) as f:
        label = str(row['label_id'])
        combined_label = str(row['combine_with_id'])
        if combined_label != '0':
            destination = prefix + '/Samples/' + combined_label
        else:
            destination = prefix + '/Samples/' + label
        Path(destination).mkdir(parents=True, exist_ok=True)
        with sf.SoundFile(f) as sf_desc:
            sr = sf_desc.samplerate
            pos = int(row['offset'] * sr)
            frame_duration = int(conf.CLIP_SECS * sr)
            try:
                sf_desc.seek(pos)
                clip_data = sf_desc.read(frames=frame_duration)
                clip_file_name = destination + '/' + label + '-' + row['file_name'] + '-' + str(row['offset']) + '.wav'
                sf.write(clip_file_name, clip_data, sr)
                return clip_file_name
            except Exception as e:
                print('Error: {e} in {fpath} at {offset}'.format(e=e, fpath=fpath, offset=row["offset"]))
                return None

# COMMAND ----------

# Build query for labels
sql_command = '''
SELECT audio_files.path, labeled_clips.file_name, labeled_clips.offset, labeled_clips.label_id, model_labels.combine_with_id 
FROM model_labels
JOIN labeled_clips ON model_labels.label_id=labeled_clips.label_id
JOIN audio_files ON labeled_clips.file_name=audio_files.name
JOIN equipment ON audio_files.sn=equipment.serial_number
JOIN monitoring_stations ON equipment.station_id=monitoring_stations.id
JOIN projects ON monitoring_stations.project_id=projects.id
JOIN model_iterations ON model_labels.iteration_id=model_iterations.id
JOIN ml_models ON model_iterations.model_id=ml_models.id
WHERE projects.id=ml_models.project_id AND model_iterations.id={iteration} AND labeled_clips.certain=TRUE
'''.format(iteration=iteration)
# Run query and store output in dataframe
label_data = pd.read_sql(sql_command, conn)

# COMMAND ----------

# Fix ids being misinterpreted as floats
label_data['combine_with_id'] = label_data['combine_with_id'].fillna(0).astype(int)

# COMMAND ----------

# Generate clips from labels
label_data['output_file'] = label_data.apply(gen_clip, axis=1)

# COMMAND ----------

# Download all external .wav files for the current row's label id
def copy_clip(row):
    label = str(row['label_id'])
    combined_label = str(row['combine_with_id'])
    if combined_label != '0':
        destination = prefix + '/Samples/' + combined_label
    else:
        destination = prefix + '/Samples/' + label
    remote_path = '/training data/uploads/' + label
    try:
        file_list = adl.ls(remote_path)
        Path(destination).mkdir(parents=True, exist_ok=True)
        count = 0
        for f in file_list:
            if Path(f).suffix == '.wav':
                file_name = Path(f).name
                local_f = destination + '/' + file_name
                adl.get(f, local_f)
#                os.symlink('../../' + local_f, 'Samples/' + label + '/' + file_name)
                count += 1
        return count
    except FileNotFoundError as e:
        return 0

# COMMAND ----------

# Build query for labels
sql_command = '''
SELECT label_id, combine_with_id
FROM model_labels
WHERE iteration_id={iteration}
'''.format(iteration=iteration)
# Run query and store output in dataframe
external_labels = pd.read_sql(sql_command, conn)

# COMMAND ----------

# Fix ids being misinterpreted as floats
external_labels['combine_with_id'] = external_labels['combine_with_id'].fillna(0).astype(int)

# COMMAND ----------

# Copy clips from Azure
external_labels['copied'] = external_labels.apply(copy_clip, axis=1)

# COMMAND ----------

# MAGIC %sh
# MAGIC 
# MAGIC # Download Panotti source code
# MAGIC git clone https://github.com/drscotthawley/panotti.git

# COMMAND ----------

# MAGIC %sh
# MAGIC 
# MAGIC # Convert .wav clips to standarized mel spectrograms, spilt into training and verification sets
# MAGIC cd iterations/$ITER
# MAGIC /databricks/driver/panotti/preprocess_data.py --resample 24000 --dur $DUR --mono

# COMMAND ----------

# MAGIC %sh
# MAGIC 
# MAGIC # Run Panotti training script
# MAGIC cd iterations/$ITER
# MAGIC /databricks/driver/panotti/train_network.py --epochs 80

# COMMAND ----------

# MAGIC %sh
# MAGIC 
# MAGIC # Run Panotti evaluation script to find training errors
# MAGIC cd iterations/$ITER
# MAGIC /databricks/driver/panotti/eval_network.py > eval.txt

# COMMAND ----------

# MAGIC %sh
# MAGIC 
# MAGIC # Save trained model
# MAGIC mkdir -p /dbfs/iterations/$ITER
# MAGIC cd iterations/$ITER
# MAGIC cp weights.hdf5 /dbfs/iterations/$ITER
# MAGIC cp eval.txt /dbfs/iterations/$ITER

# COMMAND ----------

# Regular expression to extract training errors from script output
import re

f = open('/dbfs/iterations/' + str(iteration) + '/eval.txt', 'r')
txt = f.read()

p = re.compile('/\d*-(.*\.wav)-(\d*\.\d*)\.wav\.npz *: should be (\d*) but came out as (\d*)')
matches = p.findall(txt)
errors = [(str(iteration),) + m for m in matches]

# COMMAND ----------

# Store training errors in database
from psycopg2.extras import execute_values

with conn.cursor() as curs:
    delete_query = 'delete from training_errors where iteration_id = %s'
    curs.execute(delete_query, (iteration,))
    insert_query = 'insert into training_errors (iteration_id, file_name, "offset", should_be_id, came_out_id) values %s'
    execute_values (
        curs, insert_query, errors, template=None, page_size=100
    )
    conn.commit()

# COMMAND ----------

# Compute evaluated accuracy
p2 = re.compile('Found (\d*) total mistakes out of (\d*)')
m2 = p2.findall(txt)
accuracy = round((100 - (int(m2[0][0])/int(m2[0][1])) * 100), 2)

# COMMAND ----------

# Store accuracy in database
update_accuracy_sql = 'update model_iterations set accuracy = %s where id = %s'

with conn.cursor() as curs:
    curs.execute(update_accuracy_sql, (accuracy, iteration))
    conn.commit()

# COMMAND ----------

dbutils.notebook.exit('OK')
