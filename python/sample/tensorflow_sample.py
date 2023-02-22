import os
import tensorflow as tf

# Define the training data
x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]

# Define the model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=1, input_shape=[1]))

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(0.1), loss='mean_squared_error')

# Define the file name for the saved model
model_file = 'tensorflow_sample.h5'

# Check if a saved model exists
if os.path.exists(model_file):
    # Load the saved model
    model = tf.keras.models.load_model(model_file)
else:
    # Train the model
    model.fit(x_train, y_train, epochs=1000)
    # Save the model
    model.save(model_file)

# Use the model to make predictions
print(model.predict([9, 10]))