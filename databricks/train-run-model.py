# Databricks notebook source
import os

class Config(object):
    CLIP_SECS = float(os.environ.get('CLIP_SECS') or 4)

    # Database
    DB_HOST = os.environ.get('DB_HOST')
    DB_NAME = os.environ.get('DB_NAME')
    DB_USER = os.environ.get('DB_USER')
    DB_PASSWORD = os.environ.get('DB_PASSWORD')

# COMMAND ----------

from sqlalchemy import create_engine, Column, Integer, Float, String, Enum, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from urllib.parse import quote
from datetime import datetime, timedelta
import enum


# Load config variables from environment
conf = Config()

# Database connection
postgres_str = ('postgresql://{username}:{password}@{ipaddress}:{port}/{dbname}'.format(username=conf.DB_USER, password=quote(conf.DB_PASSWORD), ipaddress=conf.DB_HOST, port=5432, dbname=conf.DB_NAME))
engine = create_engine(postgres_str)
Session = sessionmaker(bind=engine)

# COMMAND ----------

# SQLAlchemy models
Base = declarative_base()


class StatusEnum(enum.Enum):
    labeling = 'Labeling'
    ready_to_train = 'Ready to train'
    training = 'Training'
    trained = 'Trained'
    ready_to_run = 'Ready to run'
    running = 'Running'
    finished = 'Finished'

    def __str__(self):
        return self.value
      
      
class ModelIteration(Base):
    __tablename__ = 'model_iterations'
    id = Column(Integer, primary_key=True)
    model_id = Column(Integer, nullable=False)
    updated = Column(DateTime, default=datetime.utcnow, nullable=False)
    description = Column(String(255))
    status = Column(Enum(StatusEnum), default=StatusEnum.labeling)
    accuracy = Column(Float)
    

# COMMAND ----------

# Get next iteration that is in ready state
session = Session()
iteration = session.query(ModelIteration).filter((ModelIteration.status == StatusEnum.ready_to_train) | (ModelIteration.status == StatusEnum.ready_to_run)).order_by(ModelIteration.updated).first()
if iteration:
  print(iteration.id)

# COMMAND ----------

if iteration and iteration.status == StatusEnum.ready_to_train:
  iteration.status = StatusEnum.training
  iteration.updated = datetime.utcnow()
  session.commit()
  result = dbutils.notebook.run('train-panotti-model', 0, {'iteration': iteration.id})
  if result == 'OK':
    iteration.status = StatusEnum.trained
    iteration.updated = datetime.utcnow()
    session.commit()

# COMMAND ----------

if iteration and iteration.status == StatusEnum.ready_to_run:
  iteration.status = StatusEnum.running
  iteration.updated = datetime.utcnow()
  session.commit()
  result = dbutils.notebook.run('run-panotti-model', 0, {'iteration': iteration.id})
  if result == 'OK':
    iteration.status = StatusEnum.finished
    iteration.updated = datetime.utcnow()
    session.commit()
