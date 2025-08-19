import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.utils import to_categorical

# --- 1. Define Parameters ---
P = 19  # Modulus
HIDDEN_LAYER_SIZE_1 = 64
HIDDEN_LAYER_SIZE_2 = 64

# --- 2. Generate Dataset ---
# We want to learn the function (a + b) mod P for all 0 <= a, b < P.
# Create all possible pairs of (a, b)
a_values = np.arange(P)
b_values = np.arange(P)
data = np.array(np.meshgrid(a_values, b_values)).T.reshape(-1, 2)
X_raw = data
y_raw = np.sum(X_raw, axis=1) % P

print(f"Generated {len(X_raw)} training samples.")
print("Example: a=3, b=5, (a+b)%P = (3+5)%19 = 8")
idx_example = np.where((X_raw[:, 0] == 3) & (X_raw[:, 1] == 5))
print(f"Raw input: {X_raw[idx_example]}, Raw output: {y_raw[idx_example]}")
print("-" * 30)

# --- 3. Preprocess the Data (One-Hot Encoding) ---
# The input to the network for a pair (a,b) will be the concatenation
# of the one-hot vectors for a and b.
# The output will be a one-hot vector of the result.

# One-hot encode inputs
X_a_one_hot = to_categorical(X_raw[:, 0], num_classes=P)
X_b_one_hot = to_categorical(X_raw[:, 1], num_classes=P)
X_train = np.concatenate([X_a_one_hot, X_b_one_hot], axis=1)

# One-hot encode outputs
y_train = to_categorical(y_raw, num_classes=P)

print(f"Shape of the training data (X_train): {X_train.shape}")
print(f"Shape of the training labels (y_train): {y_train.shape}")
print(f"Input for (a=3, b=5) is two concatenated one-hot vectors of size {P}.")
print("-" * 30)

# --- 4. Define the MLP Model ---
model = Sequential([
    Input(shape=(2 * P,)),
    Dense(HIDDEN_LAYER_SIZE_1, activation='relu'),
    Dense(HIDDEN_LAYER_SIZE_2, activation='relu'),
    Dense(P, activation='softmax')  # Output layer with P neurons for P classes
])

# --- 5. Compile the Model ---
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()
print("-" * 30)

# --- 6. Train the Model ---
print("Starting model training...")
# Training for enough epochs to learn the function. This is a deterministic
# function, so the model should be able to achieve very high accuracy.
history = model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=32,
    verbose=1 # Set to 1 to see progress
)
print("Model training finished.")
print("-" * 30)

# --- 7. Evaluate the Model ---
# We can evaluate on the training data itself since we want the model
# to perfectly learn the entire function space.
loss, accuracy = model.evaluate(X_train, y_train, verbose=0)
print(f"Final Model Accuracy: {accuracy * 100:.2f}%")
print("-" * 30)


# --- 8. Demonstrate Predictions ---
def predict_addition(a, b):
    """Predicts the result of (a + b) mod P using the trained model."""
    if not (0 <= a < P and 0 <= b < P):
        raise ValueError(f"Inputs a and b must be between 0 and {P-1}")

    # Preprocess the inputs
    a_one_hot = to_categorical([a], num_classes=P)
    b_one_hot = to_categorical([b], num_classes=P)
    input_vector = np.concatenate([a_one_hot, b_one_hot], axis=1)

    # Make a prediction
    prediction = model.predict(input_vector)
    predicted_class = np.argmax(prediction, axis=1)[0]
    actual_result = (a + b) % P

    print(f"Input: a={a}, b={b}")
    print(f"Expected result: ({a} + {b}) % {P} = {actual_result}")
    print(f"Model's predicted result: {predicted_class}")
    print(f"Correct prediction: {predicted_class == actual_result}")
    print("-" * 10)

# Example predictions
print("Demonstrating model predictions:")
predict_addition(5, 7)
predict_addition(18, 18)
predict_addition(0, 0)
predict_addition(10, 10)
