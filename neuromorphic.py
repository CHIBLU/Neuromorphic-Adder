import numpy as np
import matplotlib.pyplot as plt

# Define neuron model (Integrate-and-Fire)
class Neuron:
    def __init__(self):
        self.membrane_potential = 0
        self.threshold = 1.0  # Threshold for firing

    def integrate(self, input_spike):
        self.membrane_potential += input_spike

    def fire(self):
        if self.membrane_potential >= self.threshold:
            self.membrane_potential = 0
            return 1  # Neuron fires
        else:
            return 0  # Neuron does not fire

# Define synapse model (with fixed weights)
class Synapse:
    def __init__(self, weight):
        self.weight = weight

    def transmit(self, spike):
        return spike * self.weight

# Generate training data (A, B, Cin, Sum, Cout)
def generate_training_data(num_samples):
    inputs = np.random.randint(0, 2, size=(num_samples, 3))  # Generate random binary inputs
    outputs = np.zeros((num_samples, 2))  # Initialize outputs

    for i in range(num_samples):
        A, B, Cin = inputs[i]
        Sum = (A ^ B) ^ Cin
        Cout = (A & B) | (B & Cin) | (A & Cin)
        outputs[i] = [Sum, Cout]

    return inputs, outputs

# Define activation function (sigmoid)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Define derivative of sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)

# Train the network using backpropagation
def train_neural_network(inputs, outputs):
    global weights_input_hidden, weights_hidden_output

    for i in range(1000):  # Perform 1000 iterations
        # Forward pass
        hidden_layer_input = np.dot(inputs, weights_input_hidden)
        hidden_layer_output = sigmoid(hidden_layer_input)

        output_layer_input = np.dot(hidden_layer_output, weights_hidden_output)
        output_layer_output = sigmoid(output_layer_input)

        # Backpropagation
        output_error = outputs - output_layer_output
        output_delta = output_error * sigmoid_derivative(output_layer_output)

        hidden_error = output_delta.dot(weights_hidden_output.T)
        hidden_delta = hidden_error * sigmoid_derivative(hidden_layer_output)

        # Update weights
        weights_hidden_output += hidden_layer_output.T.dot(output_delta) * learning_rate
        weights_input_hidden += inputs.T.dot(hidden_delta) * learning_rate

# Validate the model
def validate_model(inputs, outputs):
    hidden_layer_input = np.dot(inputs, weights_input_hidden)
    hidden_layer_output = sigmoid(hidden_layer_input)

    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output)
    output_layer_output = sigmoid(output_layer_input)

    # Compute accuracy
    predictions = np.round(output_layer_output)
    accuracy = np.mean(predictions == outputs)
    print("Validation Accuracy:", accuracy)

# Test the model with specific inputs
def test_model(input_data):
    hidden_layer_input = np.dot(input_data, weights_input_hidden)
    hidden_layer_output = sigmoid(hidden_layer_input)

    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output)
    output_layer_output = sigmoid(output_layer_input)

    return np.round(output_layer_output)[0]  # Return output as a single array instead of unpacking

# Generate test data
test_inputs = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
                        [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])

# Initialize weights randomly
weights_input_hidden = np.random.uniform(-1, 1, (3, 4))
weights_hidden_output = np.random.uniform(-1, 1, (4, 2))

# Define learning rate
learning_rate = 0.1

# Generate training data
inputs, outputs = generate_training_data(100)

# Train the neural network
train_neural_network(inputs, outputs)

# Validate the model
validate_model(inputs, outputs)

# Define the generate_output_waveforms function
def generate_output_waveforms(inputs):
    waveforms = {'A': [], 'B': [], 'Cin': [], 'Sum': [], 'Cout': []}
    for input_data in inputs:
        A, B, Cin = input_data
        waveforms['A'].append(A)
        waveforms['B'].append(B)
        waveforms['Cin'].append(Cin)

        # Simulate the neural network for current input
        output = test_model(np.array([input_data]))
        Sum, Cout = output[0], output[1]

        waveforms['Sum'].append(Sum)
        waveforms['Cout'].append(Cout)
    return waveforms

# Generate and plot output waveforms
def generate_and_plot_output_waveforms(inputs):
    waveforms = generate_output_waveforms(inputs)

    plt.figure(figsize=(10, 8))

    plt.subplot(4, 1, 1)
    plt.plot(range(len(waveforms['A'])), waveforms['A'], label='A')
    plt.plot(range(len(waveforms['B'])), waveforms['B'], label='B')
    plt.plot(range(len(waveforms['Cin'])), waveforms['Cin'], label='Cin')
    plt.xlabel('Time')
    plt.ylabel('Input')
    plt.title('Input Waveforms')
    plt.legend()

    plt.subplot(4, 1, 2)
    plt.plot(range(len(waveforms['Sum'])), waveforms['Sum'], label='Sum')
    plt.xlabel('Time')
    plt.ylabel('Output')
    plt.title('Sum Output Waveform')

    plt.subplot(4, 1, 3)
    plt.plot(range(len(waveforms['Cout'])), waveforms['Cout'], label='Cout')
    plt.xlabel('Time')
    plt.ylabel('Output')
    plt.title('Carry Out (Cout) Waveform')

    plt.tight_layout()
    plt.show()

# Generate and plot output waveforms
generate_and_plot_output_waveforms(test_inputs)
