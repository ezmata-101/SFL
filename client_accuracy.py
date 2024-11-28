import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'cloud.json'

import random
import subprocess
import pickle
import argparse
from google.cloud import storage
import tensorflow as tf
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.models import load_model

bucket_name = 'hfl-data'
storage_client = storage.Client()
X_test = None
y_test = None

def upload_to_cloud(file_name):
    try:
        bucket = storage_client.get_bucket(bucket_name)
        blob = bucket.blob(file_name)
        blob.upload_from_filename(file_name)
    except Exception as e:
        return

def download_from_cloud(file_name):
    try:
        bucket = storage_client.get_bucket(bucket_name)
        blob = bucket.blob(file_name)
        data = blob.download_as_string()
        res = pickle.loads(data)
        return res
    except Exception as e:
        return

def read_test_data():
  X_test = download_from_cloud('test_data_X.pkl')
  y_test = download_from_cloud('test_data_y.pkl')

  return X_test, y_test

def write_result(client_no, result):
  #combined_acc = {'loss': results[0], 'acc': results[1]}
  with open('client_acc_{}.pkl'.format(client_no), 'wb') as client_acc:
      pickle.dump(result, client_acc)

  upload_to_cloud(f'client_acc_{client_no}.pkl')

  return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Optional app description')

    parser.add_argument('client_no', type=str, help="The serial of client passed from server")
    args = parser.parse_args()
    client_no = args.client_no

    X_test, y_test = read_test_data()
    model = load_model('iris.keras')
    test_accuracy = tf.keras.metrics.Accuracy()
    ds_test_batch  = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(10)
    for (x, y) in ds_test_batch:
        logits = model(x, training=False)
        prediction = tf.math.argmax(logits, axis=1, output_type=tf.int64)
        test_accuracy(prediction, y)

    result = test_accuracy.result().numpy().item()
    write_result(client_no, result)
