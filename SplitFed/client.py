import tensorflow as tf
import numpy as np
import os
import time

# Simulate client-side model
def client_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(784,)),
        tf.keras.layers.Dense(128, activation='relu'),
    ])
    return model

# File paths for communication
activation_file = "intermediate_activations.npy"
gradient_file = "client_gradients.npy"

# Create client model
client = client_model()
client_optimizer = tf.keras.optimizers.Adam()

# Dummy data
(x_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 784) / 255.0
y_train = tf.keras.utils.to_categorical(y_train, 10)

batch_size = 32

# Training loop
for epoch in range(5):
    for i in range(0, len(x_train), batch_size):
        x_batch = x_train[i:i + batch_size]
        y_batch = y_train[i:i + batch_size]

        # Forward pass (client-side)
        with tf.GradientTape() as tape:
            client_output = client(x_batch)
            # Save intermediate activations
            np.save(activation_file, client_output.numpy())

        # Wait for server to process
        input("Press Enter to continue...")

        # Load gradients from file
        client_gradients = np.load(gradient_file)
        os.remove(gradient_file)  # Clean up file after reading

        # Apply gradients (client-side)
        client_grads = [tf.convert_to_tensor(client_gradients)]
        client_optimizer.apply_gradients(zip(client_grads, client.trainable_variables))

    print(f"Client Epoch {epoch + 1} completed")
