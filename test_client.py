import os
from google.cloud import storage
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'cloud.json'

import math
import random
import subprocess
import pickle
import argparse
import tensorflow as tf
import numpy as np
from numpy.polynomial import polynomial as poly
import json
from sklearn import datasets
from sklearn.model_selection import train_test_split

from tensorflow.keras import Input
from tensorflow.keras import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy

'''
Arguments:
 loss, metrics, (lr, decay, momentum): SGD, (shape, classes_count): SimpleMLP, global_weights, epochs, verbose, client_batched_data

 #Careful:
   - metrics can be an array

Return:
 gradients received from local_model
'''

PK = None
SK = None
p = 58727 # both the primes are 15 bits
q = 65063
n = None
g = None
lmbda = None
mu = None
client_no = 1
X_train = None
y_train = None
agg_gradients = None

bucket_name = 'hfl-data'
storage_client = storage.Client()
SERVER_FILE = 'agg_gradients.pkl'
batch_size = 32
loss_object = SparseCategoricalCrossentropy()

######################## ARG LIST ########################
loss = None
metrics = None
lr = None
decay = None
momentum = None
shape = None
classes_count = None
global_weights = None
epochs = None
verbose = None
client_batched_data = None
#########################################################

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

def get_compiled_model():
    '''
        returns compiled keras model
    '''

    inputs = Input(shape=(784,), name="digits")
    x1 = Dense(64, activation="relu")(inputs)
    x2 = Dense(64, activation="relu")(x1)
    outputs = Dense(10, name="predictions")(x2)
    model = Model(inputs=inputs, outputs=outputs)

    loss_fn = SparseCategoricalCrossentropy(from_logits=True)
    optimizer = Adam(learning_rate=1e-3)

    return model, loss_fn, optimizer

def read_client_batched_data(bs=32):
    global client_batched_data
    client_batched_data_list = download_from_cloud(f'client_batched_data_{client_no}.pkl')
    # with open('client_batched_data_{}.pkl'.format(client_no), 'rb') as pickle_file:
    #     client_batched_data_list = pickle.load(pickle_file)

    # Convert list of numpy array tuples back to TensorFlow _BatchDataset
    client_batched_data = None
    for data, labels in client_batched_data_list:
        dataset = tf.data.Dataset.from_tensor_slices((data, labels))
        if client_batched_data is None:
            client_batched_data = dataset
        else:
            client_batched_data = client_batched_data.concatenate(dataset)

def read_client_dataset():
    global X_train, y_train, X_test, y_test
    client_dataset = download_from_cloud(f'client_dataset_{client_no}.pkl')
    # with open('client_dataset_{}.pkl'.format(client_no), 'rb') as pickle_file:
    #     client_dataset = pickle.load(pickle_file)

    Xt, yt = zip(*client_dataset)
    X_train = np.array(Xt, dtype=np.float32)
    y_train = np.array(yt)

def tryDownloadingAgg():
    try:
        bucket = storage_client.get_bucket(bucket_name)
        blob = bucket.blob('agg_gradients.pkl')
        data = blob.download_as_string()
        globals()['agg_gradients'] = pickle.loads(data)
        return True
    except Exception as e:
        return False

def read_updated_gradients():
    # global agg_gradients
    updated_gradients = download_from_cloud(SERVER_FILE)
    # with open(SERVER_FILE, 'rb') as pickle_file:
    #     updated_gradients = pickle.load(pickle_file)

    q_agg_gradients = DEC(updated_gradients)
    globals()['agg_gradients'] = dequantize(q_agg_gradients)

    return

def write_local_accuracy(acc):
    with open(f'local_accuracy_{client_no}.pkl', 'wb') as f:
        pickle.dump(acc, f)

    upload_to_cloud(f'local_accuracy_{client_no}.pkl')

    return

def new_local_training(model, optimizer, ckpt, manager):
    epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

    grads = None
    ckpt.restore(manager.latest_checkpoint)

    for epoch in range(epochs):
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                logits = model(x_batch_train, training=True)
                loss_value = loss_fn(y_batch_train, logits)

            grads = tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            epoch_accuracy.update_state(y_batch_train, model(x_batch_train, training=True))
            ckpt.step.assign_add(1)

            if int(ckpt.step) % 100 == 0:
                save_path = manager.save()

    epoch_acc = epoch_accuracy.result().numpy().item()
    write_local_accuracy(epoch_acc)

    return grads

if __name__ == '__main__':
    # Instantiate the parser
    parser = argparse.ArgumentParser(description='Optional app description')

    parser.add_argument('loss', type=str, help="Required argument for the loss method to be used")
    parser.add_argument('metrics', type=str, help="Required argument for the metrics to be used")
    parser.add_argument('lr', type=float, help="Required argument for lr value")
    parser.add_argument('decay', type=float, help="Required argument for ratio of lr & comms_round")
    parser.add_argument('momentum', type=float, help="Required argument for ratio optimizer momentum")
    parser.add_argument('shape', type=int, help="Required argument for shape of dataset")
    parser.add_argument('classes_count', type=int, help="Required argument for total number of classes")
    parser.add_argument('epoch', type=int, help="Required argument for number of epochs")
    parser.add_argument('verbose', type=int, help="Required argument for verbose")

    args = parser.parse_args()

    loss = args.loss
    metrics = args.metrics
    lr = args.lr
    decay = args.decay
    momentum = args.momentum
    shape = args.shape
    classes_count = args.classes_count
    epochs = args.epoch
    verbose = args.verbose

    model, loss_fn, optimizer = get_compiled_model()
    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, model=model)
    manager = tf.train.CheckpointManager(ckpt, './tf_ckpts', max_to_keep=3)

    # PK, SK = read_paillier_keys()
    PK, SK = read_creds()

    #server_file_exists = os.path.isfile(SERVER_FILE)
    server_file_exists = tryDownloadingAgg()

    if server_file_exists:
        read_updated_gradients()
        globals()['agg_gradients'] = [tf.cast(grad, tf.float32) for grad in globals()['agg_gradients']]
        optimizer.apply_gradients(zip(globals()['agg_gradients'], model.trainable_variables))

    read_client_dataset()

    grads = new_local_training(model, optimizer, ckpt, manager)

    with open('local_gradients_{}.pkl'.format(client_no), 'wb') as pickle_file:
        pickle.dump(enc_grads, pickle_file)

    upload_to_cloud(f'local_gradients_{client_no}.pkl')

    model.save('iris.keras')
    print(client_no)
