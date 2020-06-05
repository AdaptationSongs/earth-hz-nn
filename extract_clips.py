import sys
import pandas as pd
import psycopg2 as pg
import soundfile as sf
from pathlib import Path
from config import Config
## Required for Azure Data Lake Storage Gen1 filesystem management
from azure.datalake.store import core, lib, multithread

# load configuration from .env file
conf = Config()


# outputs standard sized .wav files for a row from input dataframe
def gen_clip(row):
    fpath = row['path'] + '/' + row['file_name']
    with adl.open(fpath) as f:
        destination = project + '/Samples/' + row['label']
        Path(destination).mkdir(parents=True, exist_ok=True)
        with sf.SoundFile(f) as sf_desc:
            sr = sf_desc.samplerate
            pos = int(row['offset'] * sr)
            frame_duration = int(conf.CLIP_SECS * sr)
            try:
                sf_desc.seek(pos)
                clip_data = sf_desc.read(frames=frame_duration)
                clip_file_name = destination + '/' + row['file_name'] + '-' + str(row['offset']) + '.wav'
                sf.write(clip_file_name, clip_data, sr)
                return clip_file_name
            except Exception as e:
                print('Error {e} in {fpath} at {offset}'.format(e=e, fpath=fpath, offset=row["offset"]))
                return None


if __name__ == '__main__':
    # get project name from command line
    project = sys.argv[1]
    # Azure Datalake authentication
    adlCreds = lib.auth(tenant_id=conf.ADLS_TENANT, client_secret=conf.ADLS_CLIENT_SECRET, client_id=conf.ADLS_CLIENT_ID, resource='https://datalake.azure.net/')
    # Create a filesystem client object
    adl = core.AzureDLFileSystem(adlCreds, store_name=conf.ADLS_ACCOUNT)
    # connect to database
    conn_string = 'host={host} dbname={name} user={user} password={password}'.format(host=conf.DB_HOST, name=conf.DB_NAME, user=conf.DB_USER, password=conf.DB_PASSWORD)
    conn = pg.connect(conn_string)
    # build query for labels
    sql_command = '''
SELECT audio_files.path, labeled_clips.file_name, labeled_clips.offset,
labels.name AS label, sub.name AS sub_label, labeled_clips.certain, labeled_clips.notes, 
labeled_clips.modified, users.name AS user 
FROM labeled_clips JOIN audio_files ON labeled_clips.file_name=audio_files.name
JOIN labels ON labeled_clips.label_id=labels.id
LEFT JOIN labels AS sub ON labeled_clips.sub_label_id=sub.id
JOIN users ON labeled_clips.user_id=users.id
JOIN equipment ON audio_files.sn=equipment.serial_number
JOIN monitoring_stations ON equipment.station_id=monitoring_stations.id
JOIN projects ON monitoring_stations.project_id=projects.id
WHERE projects.name='{project}'
'''.format(project=project)
    # run query and store output in dataframe
    label_data = pd.read_sql(sql_command, conn)
    # generate clips from labels
    label_data['output_file'] = label_data.apply(gen_clip, axis=1)
