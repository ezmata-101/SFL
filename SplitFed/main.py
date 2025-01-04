import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
import numpy as np

# Generate synthetic data for demonstration
num_samples = 1000
# x_train = np.random.rand(num_samples, 28, 28)  # Random 28x28 images
# y_train = np.random.randint(0, 10, size=(num_samples,))  # Random labels for 10 classes

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the dataset
x_train = x_train / 255.0  # Scale pixel values to [0, 1]
x_test = x_test / 255.0


# Client-side model
input_layer = Input(shape=(28, 28))  # Input layer
flatten_layer = Flatten()(input_layer)  # Flattening the input
hidden_layer_1 = Dense(128, activation="relu")(flatten_layer)  # First hidden layer
client_model = Model(inputs=input_layer, outputs=hidden_layer_1)

# Server-side model
server_input = Input(shape=(128,))  # Input from the client-side model
hidden_layer_2 = Dense(64, activation="relu")(server_input)  # Second hidden layer
output_layer = Dense(10, activation="softmax")(hidden_layer_2)  # Output layer
server_model = Model(inputs=server_input, outputs=output_layer)

# Optimizers
client_optimizer = Adam(learning_rate=0.001)
server_optimizer = Adam(learning_rate=0.001)

# Loss function
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

# Training settings
batch_size = 32
epochs = 5 # Set this to a higher value for actual training

# Training loop
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    for step in range(0, len(x_train), batch_size):
        # Extract batch
        x_batch = x_train[step:step + batch_size]
        y_batch = y_train[step:step + batch_size]

        with tf.GradientTape(persistent=True) as tape:
            # Forward pass through client-side model
            client_output = client_model(x_batch, training=True)

            # np.save(f'client_output_{epoch}.npy', client_output)
            #
            # retrieved_client_output = np.load(f'client_output_{epoch}.npy')

            # Forward pass through server-side model
            predictions = server_model(client_output, training=True)

            # Compute loss
            loss = loss_fn(y_batch, predictions)

        # Backward pass through server-side model
        server_grads = tape.gradient(loss, server_model.trainable_weights)
        server_optimizer.apply_gradients(zip(server_grads, server_model.trainable_weights))

        # Backward pass through client-side model
        client_grads = tape.gradient(loss, client_model.trainable_weights)
        client_optimizer.apply_gradients(zip(client_grads, client_model.trainable_weights))

        # Clean up
        del tape

    # Print loss for the epoch
    print(f"Loss after epoch {epoch + 1}: {loss.numpy()}")


# Testing the combined model
def test_model(client_model, server_model, x_test, y_test):
    # Forward pass through client-side model
    client_output = client_model(x_test, training=False)
    # Forward pass through server-side model
    predictions = server_model(client_output, training=False)
    # Compute accuracy
    accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    accuracy.update_state(y_test, predictions)
    print(f"Test Accuracy: {accuracy.result().numpy()}")


# Synthetic test data for demonstration
# x_test = np.random.rand(200, 28, 28)  # Random 28x28 test images
# y_test = np.random.randint(0, 10, size=(200,))  # Random test labels

test_model(client_model, server_model, x_test, y_test)
