import os
import traceback
import glob
import math
from google.cloud import storage
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'cloud.json'

import random
import subprocess
import pickle
from datetime import datetime
import tensorflow as tf
import numpy as np
from numpy.polynomial import polynomial as poly
from tqdm import tqdm

from sklearn import datasets
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras import backend as K

# Define hyperparameters
SYNC_MODE = True
lr = 0.01
comms_round = 32
percentile = 12.5 # For Sync-Async switch
loss='sparse_categorical_crossentropy'
metrics = ['sparse_categorical_accuracy'] #['accuracy']
client_epoch = 1
client_verbose = 0
decay = lr / comms_round #learning rate decay
momentum = 0.9
classes_count = 3
shape = None

bucket_name = 'hfl-data'
storage_client = storage.Client()
data_path = './swarm_aligned'
clients = None
clients_batched = None
PK = None
ASYNC_TIMEOUT = 120 # 2 minutes
lock = threading.Lock()
comp_time = 0

n = 2**4
q = 2**511
t = 2**63
poly_mod = np.array([1] + [0] * (n - 1) + [1])

# Data Structures
client_private_ips = [
    '10.128.0.3',
    '10.128.0.4',
    '10.128.0.5',
    '10.128.0.6'
]

client_public_ips = [
    '104.197.89.108',
    '34.41.69.242',
    '34.30.32.19',
    '34.27.85.124'
]

global_gradient = None

latest_client_round = {
    1: 0, # client_no: latest_round
    2: 0,
    3: 0,
    4: 0
}

latest_client_grad = {
    1: None, # client_no: latest_grad
    2: None,
    3: None,
    4: None
}

ratios = [ 0.25, 0.25, 0.25, 0.25 ]

def upload_to_cloud(file_name):
    start = datetime.now()
    try:
        bucket = storage_client.get_bucket(bucket_name)
        blob = bucket.blob(file_name)
        blob.upload_from_filename(file_name)
        end = datetime.now()
        globals()['comp_time'] += (end-start).total_seconds()
    except Exception as e:
        end = datetime.now()
        globals()['comp_time'] += (end-start).total_seconds()
        return

def download_from_cloud(file_name):
    start = datetime.now()
    try:
        bucket = storage_client.get_bucket(bucket_name)
        blob = bucket.blob(file_name)
        data = blob.download_as_string()
        res = pickle.loads(data)
        end = datetime.now()
        globals()['comp_time'] += (end-start).total_seconds()
        return res
    except Exception as e:
        end = datetime.now()
        globals()['comp_time'] += (end-start).total_seconds()
        return

def load_dataset():
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

    X_train = np.reshape(X_train, (-1, 784))
    X_test = np.reshape(X_test, (-1, 784))

    return X_train, X_test, y_train, y_test

def create_clients(data_list, label_list, num_clients=10, initial='clients'):
    client_names = ['{}_{}'.format(initial, i + 1) for i in range(num_clients)]
    data = list(zip(data_list, label_list))
    random.shuffle(data)

    # Define unique ratios for each client
    #ratios = np.random.dirichlet(np.ones(num_clients), size=1).tolist()[0]
    # ratios = [0.25, 0.25, 0.25, 0.25]
    print('-------------------------------------')
    print(f"Ratios of divided datasets: {globals()['ratios']}")
    print('-------------------------------------')

    sizes = []
    total_samples = len(data)
    remaining_samples = total_samples

    for i in range(num_clients - 1):
        size = int(globals()['ratios'][i] * total_samples)
        sizes.append(size)
        remaining_samples -= size

    sizes.append(remaining_samples)

    shards = [data[i:i + sizes[idx]] for idx, i in enumerate(np.cumsum([0] + sizes[:-1]))]

    return {client_names[i]: shards[i] for i in range(len(client_names))}

def batch_data(data_shard, bs=32):
    '''Takes in a clients data shard and create a tfds object off it
    args:
        shard: a data, label constituting a client's data shard
        bs:batch size
    return:
        tfds object'''
    #seperate shard into data and labels lists
    data, label = zip(*data_shard)
    dataset = tf.data.Dataset.from_tensor_slices((list(data), list(label)))
    return dataset.shuffle(len(label)).batch(bs)

def save_client_batched_data(client_no, client_batched_data):
    # Convert TensorFlow _BatchDataset to list of numpy arrays
    client_batched_data_list = []
    for data, labels in client_batched_data:
        client_batched_data_list.append((data.numpy(), labels.numpy()))

    with open('client_batched_data_{}.pkl'.format(client_no), 'wb') as pickle_file:
        pickle.dump(client_batched_data_list, pickle_file)

    return

def save_client_dataset(client_no, client_dataset):
    with open('client_dataset_{}.pkl'.format(client_no), 'wb') as pickle_file:
        pickle.dump(client_dataset, pickle_file)

    return

def save_agg_gradients(agg_gradients):
    with open('agg_gradients.pkl', 'wb') as pickle_file:
        pickle.dump(agg_gradients, pickle_file)

    return

def dispatch_data():
    '''
        sends data to clients
    '''

    # Save dataset files
    client_no = 1
    for client in clients:
        client_dataset = clients[client]
        save_client_dataset(client_no, client_dataset)
        client_no += 1

    # Send dataset files to clients
    for i in range(len(client_public_ips)):
        upload_to_cloud(f'client_dataset_{i+1}.pkl')
        # public_ip = client_public_ips[i]
        # scp_command = f'scp -i ~/.ssh/id_rsa ./client_dataset_{i+1}.pkl g1805021@{public_ip}:~'
        # subprocess.check_output(scp_command, shell=True, text=True).strip()

    return

def get_client_gradients(client_no, public_ip):
    # scp_command = f'scp -i ~/.ssh/id_rsa g1805021@{public_ip}:/home/g1805021/local_gradients_{client_no}.pkl /home/g1805021/FedServer/'
    # subprocess.check_output(scp_command, shell=True, text=True).strip()
    local_gradients = download_from_cloud(f'local_gradients_{client_no}.pkl')

    # with open('local_gradients_{}.pkl'.format(client_no), 'rb') as pickle_file:
    #     local_gradients = pickle.load(pickle_file)

    return local_gradients

def get_client_local_accuracy(client_no, public_ip):
    # scp_command = f'scp -i ~/.ssh/id_rsa g1805021@{public_ip}:/home/g1805021/local_accuracy_{client_no}.pkl /home/g1805021/FedServer/LA'
    # subprocess.check_output(scp_command, shell=True, text=True).strip()
    local_accuracy = download_from_cloud(f'local_accuracy_{client_no}.pkl')

    # with open('LA/local_accuracy_{}.pkl'.format(client_no), 'rb') as pickle_file:
    #     local_accuracy = pickle.load(pickle_file)

    return local_accuracy

def aggregate_client_gradients(client_grads_list, comm_round):
    '''
        returns aggregated gradients
    '''
    print('In aggregate gradients...')
    agg_gradients = []
    steps = len(client_grads_list[0])

    client_weights = None
    for i in range(steps):
        agg_gradients.append(np.mean([client_grads[i] for client_grads in client_grads_list], axis=0))

    return agg_gradients

def send_updated_gradients(grads):
    save_agg_gradients(grads)
    upload_to_cloud(f'agg_gradients.pkl')

    # for i in range(len(client_public_ips)):
        # public_ip = client_public_ips[i]
        # scp_command = f'scp -i ~/.ssh/id_rsa ./agg_gradients.pkl g1805021@{public_ip}:~'
        # subprocess.check_output(scp_command, shell=True, text=True).strip()
    return

def new_global_training():
    '''
        returns global model trained with client models
    '''

    global SYNC_MODE
    global ASYNC_TIMEOUT
    global global_gradient

    # Send the clients their respective data shards
    start = datetime.now()
    dispatch_data()
    end = datetime.now()
    print(f'Dataset distribution time: {(end-start).total_seconds()} seconds')
    globals()['comp_time'] += (end-start).total_seconds()

    # Synchronous rounds
    for comm_round in range(comms_round):
        client_no = 1
        client_grads_list = []
        client_local_acc = []
        client_names= list(clients.keys())
        random.shuffle(client_names)

        tt = 0
        for client in tqdm(client_names, desc = 'Clients completed'):
            private_ip = client_private_ips[client_no - 1]
            public_ip = client_public_ips[client_no - 1]

            start = datetime.now()
            ssh_command = f'ssh uncleroger@{private_ip} "python3 test_client.py {loss} {metrics} {lr} {decay} {momentum} {shape} {classes_count} {client_epoch} {client_verbose}"'
            result = subprocess.check_output(ssh_command, shell=True, text=True).strip() #the result contains the local model weights
            end = datetime.now()
            client_tt = (end-start).total_seconds()
            tt += client_tt
            print(f'Client {client_no} took: {client_tt} seconds')

            client_grads = get_client_gradients(client_no, public_ip)
            local_accuracy = get_client_local_accuracy(client_no, public_ip)

            client_grads_list.append(client_grads)
            client_local_acc.append(local_accuracy)
            latest_client_round[client_no] += 1
            latest_client_grad[client_no] = client_grads

            client_no += 1
            K.clear_session()

        start = datetime.now()
        print('Aggregating gradients in sync mode...')
        res = aggregate_client_gradients(client_grads_list, comm_round)
        global_gradient = res

        # Because the weights were multiplied by 100 for convenience
        if comm_round != 0:
            for i, arr in enumerate(res):
                res[i] = arr/100

        send_updated_gradients(res)
        end = datetime.now()
        globals()['comp_time'] += (end-start).total_seconds()
        avg_local_acc = sum(client_local_acc)/len(client_local_acc)

        print('----------------------------')
        print('Round {} completed'.format(comm_round+1))
        print('Average local accuracy: {}'.format(avg_local_acc))

    return

def save_test_data(X_test, y_test):
    with open('test_data_X.pkl', 'wb') as pickle_file:
        pickle.dump(X_test, pickle_file)

    with open('test_data_y.pkl', 'wb') as pickle_file:
        pickle.dump(y_test, pickle_file)

    return

def send_test_data(public_ip):
    # scp_command = f'scp -i ~/.ssh/id_rsa ./test_data_X.pkl g1805021@{public_ip}:~'
    # subprocess.check_output(scp_command, shell=True, text=True).strip()
    # scp_command = f'scp -i ~/.ssh/id_rsa ./test_data_y.pkl g1805021@{public_ip}:~'
    # subprocess.check_output(scp_command, shell=True, text=True).strip()
    upload_to_cloud(f'test_data_X.pkl')
    upload_to_cloud(f'test_data_y.pkl')

    return

def get_client_loss_accuracy(client_no, public_ip):
    result = download_from_cloud(f'client_acc_{client_no}.pkl')
    # scp_command = f'scp -i ~/.ssh/id_rsa g1805021@{public_ip}:/home/g1805021/client_acc_{client_no}.pkl /home/g1805021/FedServer/'
    # subprocess.check_output(scp_command, shell=True, text=True).strip()

    # with open('client_acc_{}.pkl'.format(client_no), 'rb') as client_acc:
    #     result = pickle.load(client_acc)

    #return combined_data['loss'], combined_data['acc']
    return result

def test_accuracy(X_test, y_test):
    client_no = 1
    loss_arr = []
    acc_arr = []
    client_names = list(clients.keys())
    random.shuffle(client_names)

    save_test_data(X_test, y_test)
    start = datetime.now()

    for client in tqdm(client_names, desc='Progress Bar'):
        private_ip = client_private_ips[client_no - 1]
        public_ip = client_public_ips[client_no - 1]

        send_test_data(public_ip)

        ssh_command = f'ssh uncleroger@{private_ip} "python3 client_accuracy.py {client_no}"'
        subprocess.check_output(ssh_command, shell=True, text=True).strip()

        acc = get_client_loss_accuracy(client_no, public_ip)
        #loss_arr.append(loss)
        acc_arr.append(acc)
        client_no += 1

        K.clear_session()

    #avg_loss = sum(loss_arr)/len(loss_arr)
    end = datetime.now()
    globals()['comp_time'] += (end-start).total_seconds()
    avg_acc = sum(acc_arr)/len(acc_arr)
    testing_time = (end-start).total_seconds()

    with open('result_avg_acc.txt', 'w') as f:
        f.write(str(avg_acc) + '\n')

    with open('result_testing_time.txt', 'w') as t:
        t.write(str(testing_time) + '\n')

    print('\n-------------------------------------------------------------')
    print(f'Average accuracy is {avg_acc}')
    print(f'Testing time taken: {testing_time}')
    print('-------------------------------------------------------------')

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_dataset()
    clients = create_clients(X_train, y_train, num_clients=4, initial='client')

    shape = X_train.shape[1]

    start = datetime.now()
    new_global_training()
    end = datetime.now()

    training_time = (end-start).total_seconds()

    with open('result_training_time.txt', 'w') as f:
        f.write(str(training_time) + '\n')

    print('----------------------------')
    print('Global training completed...')
    print(f'Training time taken: {training_time}')
    print('----------------------------')
    print('Testing accuracy')
    print('----------------------------')

    test_accuracy(X_test, y_test)
    print(f"Total computational overhead: {globals()['comp_time']}")
