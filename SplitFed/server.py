import tensorflow as tf
import numpy as np
import os
import time

# Simulate server-side model
def server_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(128,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax'),
    ])
    return model

# File paths for communication
activation_file = "intermediate_activations.npy"
gradient_file = "client_gradients.npy"

# Create server model
server = server_model()
server_optimizer = tf.keras.optimizers.Adam()

# Dummy data
(x_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()
y_train = tf.keras.utils.to_categorical(y_train, 10)

batch_size = 32

# Training loop
for epoch in range(5):
    for i in range(0, len(x_train), batch_size):
        # Wait for client intermediate activations
        input("Press Enter to continue...")

        # Load activations
        client_output = np.load(activation_file)
        os.remove(activation_file)  # Clean up file after reading

        # Forward pass (server-side)
        with tf.GradientTape() as tape:
            server_output = server(client_output)
            loss = tf.keras.losses.categorical_crossentropy(
                y_train[i:i + batch_size], server_output
            )

        # Backward pass (server-side)
        server_grads = tape.gradient(loss, server.trainable_variables)
        server_optimizer.apply_gradients(zip(server_grads, server.trainable_variables))

        # Compute gradients for the client-side model
        intermediate_grads = tape.gradient(loss, client_output)

        # Save gradients to file
        np.save(gradient_file, intermediate_grads.numpy())

    print(f"Server Epoch {epoch + 1} completed")
