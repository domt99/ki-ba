from pathlib import Path

from dotenv import dotenv_values

from ki_ba.config import Config


class Framework:
    def __init__(self):
        self.config = self.read_env_file()

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

    def validate_config(self, config, data_prep):
        ...

    def create_data_prep(self, data_set_type):
        ...

    def load_mnist(self):
        ...

    def create_calculation(self, calculation_type):
        ...

    def create_activation(self, activation_name):
        ...

    def create_layers(self, topology, activations, calculation, seed):
        ...

    def create_optimizer(self, optimizer_type):
        ...

    def train(self):
        ...

    def train_step(self, image_batch, label_batch):
        ...
