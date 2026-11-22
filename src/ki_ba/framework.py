import io
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from dotenv import dotenv_values
from PIL import Image

from ki_ba.activation_function import Identity, ReLU, SigmoidLog, SigmoidTanh
from ki_ba.calculation import CalculateLoop, CalculateMatrix
from ki_ba.config import Config
from ki_ba.dataprep import DataPrep
from ki_ba.layer import HiddenLayer, InputLayer, OutputLayer
from ki_ba.optimizer import Adam, GradientDescent, Rprop
from ki_ba.mlp import MLP


class Framework:
    def __init__(self):
        self.config = self.read_env_file()
        self.data_prep = self.create_data_prep(self.config.data_set_type)
        self.validate_config(self.config, self.data_prep)
        input_layer, hidden_layers, output_layer = self.create_layers(
            self.config.topology,
            self.config.activations,
            self.create_calculation(self.config.calculation_type),
            self.config.seed
        )
        self.mlp = MLP(input_layer, hidden_layers, output_layer)
        self.optimizer = self.create_optimizer(self.config.optimizer_type)
        self.epoch_errors = []

    def read_env_file(self, env_path=".env", template_folder="configs"):
        env_path = Path(env_path)
        values = dotenv_values(env_path)

        template = values.get("TEMPLATE")
        if template:
            template_path = env_path.parent / template_folder / template
            if not template_path.is_file():
                raise FileNotFoundError(f"Template {template_path} not found")
            values = dotenv_values(template_path)

        topology = [int(n) for n in self._require(values, "TOPOLOGY").split(",")]
        if len(topology) < 3:
            raise ValueError(
                f"TOPOLOGY needs at least 3 layers (input, hidden, output), got {len(topology)}"
            )
        activations = [a.strip().lower() for a in self._require(values, "ACTIVATIONS").split(",")]
        hidden_layer_count = len(topology) - 2
        if len(activations) == 1:
            activations = activations * hidden_layer_count
        elif len(activations) != hidden_layer_count:
            raise ValueError(
                f"ACTIVATIONS has {len(activations)} entries, "
                f"but TOPOLOGY has {hidden_layer_count} hidden layers"
            )

        return Config(
            data_set_type=self._require(values, "DATASET_TYPE").strip().lower(),
            topology=topology,
            activations=activations,
            calculation_type=self._require(values, "CALCULATION_TYPE").strip().lower(),
            optimizer_type=self._require(values, "OPTIMIZER_TYPE").strip().lower(),
            epoch_count=int(self._require(values, "EPOCH_COUNT")),
            batch_size=int(self._require(values, "BATCH_SIZE")),
            seed=int(self._require(values, "SEED")),
            learning_rate=self._optional_float(values, "LEARNING_RATE"),
        )

    def validate_config(self, config, data_prep):
        input_count = data_prep.train_images.shape[1]
        if config.topology[0] != input_count:
            raise ValueError(
                f"TOPOLOGY starts with {config.topology[0]} inputs, "
                f"but the data has {input_count} columns"
            )

        if config.topology[-1] != data_prep.class_count:
            raise ValueError(
                f"TOPOLOGY ends with {config.topology[-1]} outputs, "
                f"but the data has {data_prep.class_count} classes"
            )

        train_count = len(data_prep.train_images)
        if config.optimizer_type == "rprop" and config.batch_size != train_count:
            raise ValueError(
                f"Rprop needs full batch: BATCH_SIZE must be {train_count}, "
                f"got {config.batch_size}"
            )

    def create_data_prep(self, data_set_type):
        match data_set_type:
            case "mnist":
                raw_images, raw_labels, raw_test_images, raw_test_labels = self.load_mnist()
                return DataPrep(
                    raw_images=raw_images,
                    raw_labels=raw_labels,
                    raw_test_images=raw_test_images,
                    raw_test_labels=raw_test_labels,
                    class_count=10,
                    scale_factor=255,
                    validation_size=10000,
                    seed=self.config.seed
                )

            case "xor":
                raw_images = np.array(((0, 0), (1, 1), (1, 0), (0, 1)))
                raw_labels = np.array((0, 0, 1, 1))
                return DataPrep(
                    raw_images=raw_images,
                    raw_labels=raw_labels,
                    class_count=2,
                    scale_factor=1,
                    validation_size=0,
                    seed=self.config.seed
                )

            case _:
                raise ValueError(
                    f"Unknown DATASET_TYPE '{data_set_type}', allowed: 'xor', 'mnist'"
                )

    def load_mnist(self, data_folder="data"):
        data_folder = Path(data_folder)
        raw_images, raw_labels = self._read_mnist_parquet(
            data_folder / "train-00000-of-00001.parquet"
        )
        raw_test_images, raw_test_labels = self._read_mnist_parquet(
            data_folder / "test-00000-of-00001.parquet"
        )
        return raw_images, raw_labels, raw_test_images, raw_test_labels

    def _read_mnist_parquet(self, path):
        if not path.is_file():
            raise FileNotFoundError(f"MNIST file {path} not found")

        table = pq.read_table(path, columns=["image", "label"])
        labels = table.column("label").to_numpy()

        # each image is stored as PNG bytes; decode to a 28x28 uint8 array
        images = np.array([
            np.array(Image.open(io.BytesIO(image["bytes"])))
            for image in table.column("image").to_pylist()
        ])
        return images, labels

    def create_calculation(self, calculation_type):
        match calculation_type:
            case "matrix":
                return CalculateMatrix()

            case "loop":
                return CalculateLoop()

            case _:
                raise ValueError(
                    f"Unknown CALCULATION_TYPE '{calculation_type}', "
                    f"allowed: 'loop', 'matrix'"
                )

    def create_activation(self, activation_name):
        match activation_name:

            case "relu":
                return ReLU()

            case "sigmoid_tanh":
                return SigmoidTanh()

            case "sigmoid_log":
                return SigmoidLog()

            case "identity":
                return Identity()

            case _:
                raise ValueError(
                    f"Unknown activation '{activation_name}' in ACTIVATIONS, "
                    f"allowed: 'relu', 'sigmoid_log', 'sigmoid_tanh', 'identity'"
                )

    def create_layers(self, topology, activations, calculation, seed):

        if any(n <= 0 for n in topology):
            raise ValueError(f"All layers in TOPOLOGY need at least 1 neuron, got {topology}")
        
        weight_rng = np.random.default_rng(seed)
        hidden_layers = []

        input_layer = InputLayer(topology[0])

        for i in range(1, len(topology) - 1):
            hidden_layers.append(HiddenLayer(
                calculation=calculation,
                neuron_count_prev=topology[i - 1],
                neuron_count=topology[i],
                weight_rng=weight_rng,
                activation_func=self.create_activation(activations[i - 1])
            ))

        output_layer = OutputLayer(
            calculation=calculation,
            neuron_count_prev=topology[-2],
            neuron_count=topology[-1],
            weight_rng=weight_rng
        )

        return input_layer, hidden_layers, output_layer

    def create_optimizer(self, optimizer_type):

        learning_rate = self.config.learning_rate

        if optimizer_type == "rprop" and learning_rate is not None:
            raise ValueError("Rprop does not use LEARNING_RATE, remove it from the env file")

        if optimizer_type != "rprop" and learning_rate is not None and learning_rate <= 0:
            raise ValueError(f"LEARNING_RATE must be > 0, got {learning_rate}")
    
        match optimizer_type:
            case "gradient_descent":
                if learning_rate is None:
                    return GradientDescent()
                else:
                    return GradientDescent(learning_rate=learning_rate)

            case "rprop":
                return Rprop()

            case "adam":
                if learning_rate is None:
                    return Adam()
                else:
                    return Adam(alpha=learning_rate)

            case _:
                raise ValueError(
                    f"Unknown OPTIMIZER_TYPE '{optimizer_type}', "
                    f"allowed: 'gradient_descent', 'rprop', 'adam'"
                )

    def train(self):
        batch_size = self.config.batch_size
        for epoch in range(0, self.config.epoch_count):
            mini_batches = self.data_prep.get_mini_batches(batch_size)
            sum_batch_errors = 0
            for image_batch, label_batch in mini_batches:
                sum_batch_errors = sum_batch_errors + self.train_step(image_batch=image_batch, label_batch=label_batch)
            self.epoch_errors.append(sum_batch_errors / len(mini_batches))

        return self.epoch_errors


    def train_step(self, image_batch, label_batch):
        prediction = self.mlp.forward_propagation(image_batch)
        loss = self.mlp.calculate_loss(prediction=prediction, target_value=label_batch)
        gradients = self.mlp.backward_propagation(label_batch)
        curr_weights = self.mlp.get_weights()
        new_weights = self.optimizer.optimize(gradients=gradients, weights_old=curr_weights)
        self.mlp.set_weights(new_weights=new_weights)
        return loss

    def _require(self, values, key):
        value = values.get(key)
        if not value:
            raise ValueError(f"{key} is missing in the env file")
        return value

    def _optional_float(self, values, key):
        value = values.get(key)
        if not value:
            return None
        return float(value)
    