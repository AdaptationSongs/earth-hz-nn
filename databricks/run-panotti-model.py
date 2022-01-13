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

import pandas as pd
from pyspark.sql.functions import col, pandas_udf, explode, from_json, lit
from pyspark.sql.types import StringType, FloatType, IntegerType, ArrayType, StructType, StructField
import numpy as np
from keras.models import  load_model
from panotti.models import *
from panotti.datautils import *
import soundfile as sf
import librosa
import io
import json
from sqlalchemy import create_engine
from urllib.parse import quote

## Required for Azure Data Lake Storage Gen1 filesystem management
from azure.datalake.store import core, lib, multithread

# Load config variables from environment
conf = Config()

# Azure Datalake authentication
adlCreds = lib.auth(tenant_id=conf.ADLS_TENANT, client_secret=conf.ADLS_CLIENT_SECRET, client_id=conf.ADLS_CLIENT_ID, resource='https://datalake.azure.net/')
# Create a filesystem client object
adl = core.AzureDLFileSystem(adlCreds, store_name=conf.ADLS_ACCOUNT)

# Database connection
postgres_str = ('postgresql://{username}:{password}@{ipaddress}:{port}/{dbname}'.format(username=conf.DB_USER, password=quote(conf.DB_PASSWORD), ipaddress=conf.DB_HOST, port=5432, dbname=conf.DB_NAME))
conn = create_engine(postgres_str)

# COMMAND ----------

# new signal rate for resampling
new_sr = 24000
# size of time window to analyze in seconds
window_size = conf.CLIP_SECS
# minimum prediction probability
threshold = 0.5
# cores to use to run the model
run_cores = 200
# cores to use to write to the database
db_cores = 8

# COMMAND ----------

# Get all audio files for this project
file_query = '''
SELECT audio_files.path, audio_files.name as file_name
FROM audio_files 
JOIN equipment ON audio_files.sn=equipment.serial_number
JOIN monitoring_stations ON equipment.station_id=monitoring_stations.id
JOIN projects ON monitoring_stations.project_id=projects.id
JOIN ml_models ON projects.id=ml_models.project_id
JOIN model_iterations ON ml_models.id=model_iterations.model_id
WHERE model_iterations.id={iteration}
'''.format(iteration=iteration)
file_df = pd.read_sql_query(file_query, con=conn)

# COMMAND ----------

# Remove any duplicate files
wav_df = file_df.groupby('file_name').first().reset_index()

# COMMAND ----------

# Generate full path to files
wav_df['name'] = wav_df['path'] + '/' + wav_df['file_name']

# COMMAND ----------

# Inputs: path to file on Azure Data Lake, trained model, list of classes
# Outputs: Scores for each label for each 5 second time window, as a JSON string
def classify_file(fpath, model, classes):
    # Set some defaults
    total_load = 1
    load_count = 0
    batch_size = 1

    results = []
    with adl.open(fpath) as f:
        try:
            with sf.SoundFile(f) as sf_desc:
                sr = sf_desc.samplerate
                frame_duration = int(window_size * sr)
                offset = 0
                for clip_data in sf_desc.blocks(blocksize=frame_duration):
                    # Convert to mono
                    clip_data = np.mean(clip_data, axis=1, keepdims=True)
                    signal = librosa.resample(clip_data.T, sr, new_sr)
                    melgram = make_layered_melgram(signal, new_sr, mels=96, phase=False)
                    X = np.zeros((total_load, melgram.shape[1], melgram.shape[2], melgram.shape[3]))
                    use_len = min(X.shape[2], melgram.shape[2])
                    X[load_count,:,0:use_len] = melgram[:,:,0:use_len]
                    num_pred = X.shape[0]
                    try:
                        y_scores = model.predict(X[0:num_pred,:,:,:],batch_size=batch_size)
                        scores_list = [{'label_id': int(classes[i]), 'probability': y_scores.tolist()[0][i]} for i in range(0, len(classes))]
                    except:
                        # Unexpected clip size, fill in 0.0 for all scores
                        scores_list = [{'label_id': classes[i], 'probability': 0.0} for i in range(0, len(classes))]
                    results.append({'offset': offset, 'scores': scores_list})
                    offset += window_size
        except:
            # Corrupt wav file, return empty set
            results = []
    return json.dumps(results)

# Outputs a User Definied Function, which can be called on a column in a Spark dataframe
def classify_file_udf(model, classes):
    return pandas_udf(lambda d: d.apply(classify_file, model=model, classes=classes), returnType=StringType())

# COMMAND ----------

# Load the saved model from local file system
weights_file_name = '/dbfs/iterations/' + str(iteration) + '/weights.hdf5'
my_model, class_names = load_model_ext(weights_file_name)

# COMMAND ----------

# Convert our Pandas dataframe to Spark, partition it based on how many threads it should be processed in
spark_df = spark.createDataFrame(wav_df)
spark_df = spark_df.repartition(run_cores)

# COMMAND ----------

# Run our user defined function on each WAV file name in the dataframe
spark_df = spark_df.select('*', classify_file_udf(my_model, class_names)(col('name')).alias('results_json'))

# COMMAND ----------

# This is the slow part
spark.conf.set("spark.sql.legacy.allowCreatingManagedTableUsingNonemptyLocation","true")
spark_df.write.saveAsTable('tmp_output', mode='overwrite')

# COMMAND ----------

# Create dataframe from saved table
processed_df = spark.sql('select * from tmp_output')

# COMMAND ----------

# Explode json into one row for each offset and label
results_schema = ArrayType(StructType([StructField('offset', FloatType()), StructField('scores', ArrayType(StructType([StructField('label_id', IntegerType()), StructField('probability', FloatType())])))]))
exploded_df = processed_df.withColumn('results', explode(from_json('results_json', results_schema)))\
    .withColumn('offset', col('results.offset'))\
    .withColumn('scores', explode(col('results.scores')))\
    .withColumn('label_id', col('scores.label_id'))\
    .withColumn('probability', col('scores.probability'))

# COMMAND ----------

# Remove results below threshold
filtered_df = exploded_df.filter(exploded_df.probability >= threshold)

# COMMAND ----------

# Output format suitable for import into Earth-Hz
output_df = filtered_df\
    .withColumn('duration', lit(window_size))\
    .withColumn('iteration_id', lit(iteration))\
    .select('iteration_id', 'file_name', 'offset', 'duration', 'label_id', 'probability')

# COMMAND ----------

# Write results to database in parallel using Spark
jdbc_url = 'jdbc:postgresql://{host}/{db_name}'.format(host=conf.DB_HOST, db_name=conf.DB_NAME)
connection_props = {
  'driver': 'org.postgresql.Driver',
  'numPartitions': str(db_cores),
  'user': conf.DB_USER,
  'password': conf.DB_PASSWORD
}

output_df.write.jdbc(jdbc_url, 'model_outputs', 'append', connection_props)

# COMMAND ----------

dbutils.notebook.exit('OK')
