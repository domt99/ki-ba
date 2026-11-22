
class MLP:
    def __init__(self, input_layer, hidden_layers, output_layer):
        self.input_layer = input_layer
        self.hidden_layers = hidden_layers
        self.output_layer = output_layer

    def forward_propagation(self, current_batch):

        neuron_values = self.input_layer.forward_step(current_batch)

        for hidden_layer in self.hidden_layers:
            neuron_values = hidden_layer.forward_step(neuron_values)

        neuron_values = self.output_layer.forward_step(neuron_values)

        return neuron_values
    
    def calculate_loss(self, prediction, target_value):
        return self.output_layer.categorical_cross_entropy(prediction, target_value)

    def backward_propagation(self, target_value):
        gradients = []

        gradient, backprop_error = self.output_layer.backward_step(target_value)
        gradients.append(gradient)

        for hidden_layer in reversed(self.hidden_layers):
            gradient, backprop_error = hidden_layer.backward_step(backprop_error)
            gradients.append(gradient)

        gradients.reverse()
        return gradients

    def get_weights(self):
        weights = []

        for hidden_layer in self.hidden_layers:
            weights.append(hidden_layer.get_weights())

        weights.append(self.output_layer.get_weights())

        return weights

    def set_weights(self, new_weights):
        if len(new_weights) != len(self.hidden_layers) + 1:
            raise ValueError("New weights have the wrong dimension to be set")

        for i in range(0, len(new_weights) - 1):
            self.hidden_layers[i].set_weights(new_weights[i])

        self.output_layer.set_weights(new_weights[-1])