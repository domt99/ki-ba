# MLP architecture (milestone 2)

How our multilayer perceptron is put together and how one training run flows through it.
Based on the class diagram we're drawing in Excalidraw.

| File | What it is |
|------|------------|
| [`mlp-architecture.excalidraw`](mlp-architecture.excalidraw) | **Source** — open on [excalidraw.com](https://excalidraw.com) (☰ → Open) and edit there |
| [`mlp-architecture.svg`](mlp-architecture.svg) | Read-only export so GitHub can show it (re-export after every change, see [below](#updating-the-diagram)) |

![MLP class diagram](mlp-architecture.svg)

**Legend** (as in the diagram): dashed line + hollow triangle = *implements (interface)*, solid line +
hollow triangle = *inherits from (class)*, plain arrow = *uses*, diamond = *consists of*.
**Green = done** (designed): all components, including `Framework`, `Config` and `DataPrep`.

---

## 1. Components at a glance

| # | Component | Role | Key methods |
|---|-----------|------|-------------|
| 1 | **Framework** (`main`) | Reads + validates the config, builds everything via factory methods, runs training | `readEnvFile()`, `validateConfig()`, `createDataPrep()`, `createCalculation()`, `createLayers()`, `createOptimizer()`, `train()`, `trainStep()` |
| 2 | **Config** «dataclass» | All settings of one run | fields `dataSetType`, `topology`, `activations`, `calculationType`, `optimizerType`, `epochCount`, `batchSize`, `seed`, `learningRate`, `alpha` |
| 3 | **DataPrep** | Scales the raw data, one-hot encodes labels, splits off validation data, delivers shuffled mini-batches | `_oneHot()`, `getMiniBatches(batchSize)`, `getValidationData()` |
| 4 | **MLP** | The network: holds the layers, runs forward/backward passes over all of them | `forwardPropagation()`, `calculateLoss()`, `backwardPropagation()`, `getWeights()`, `setWeights()` |
| 5 | **Layer** «interface» | What every layer can do | `forwardStep(neuronValuesPrev)` |
| 6 | **InputLayer** | Passes the input through, computes nothing | `forwardStep()` |
| 7 | **DenseLayer** «abstract» | Shared logic of all layers with weights | `initializeWeights()`, `forwardStep()`, `backwardStep()`, abstract `activate()` / `calculateDelta()` |
| 8 | **HiddenLayer** | Dense layer with a per-layer activation function | `activate()`, `calculateDelta(backpropError)` |
| 9 | **OutputLayer** | Dense layer with softmax + categorical cross-entropy (CCE) | `softmax()`, `categoricalCrossEntropy()`, `calculateDelta(targetValue)` |
| 10 | **ActivationFunction** «interface» | `ReLU`, `Sigmoid`, `Identity` | `activationFunc(net)`, `activationFuncDerivative(net)` |
| 11 | **Calculation** «interface» | *How* the math runs: `Loop` (explicit loops) or `Matrix` (NumPy matrix ops) | `calculateNet()`, `calculateGradient()`, `calculateBackpropError()` |
| 12 | **Optimizer** «interface» | Updates the weights from the gradients: `GradientDescent`, `Rprop`, `Adam` | `optimize(g, w_old) → w_curr` |

Three places where we swap in implementations (strategy pattern), each picked by a factory method in the
Framework from a string in the `Config`:

- **Calculation** (`Loop` vs `Matrix`): same results, different implementation. Chosen once via `calculationType`
  and passed to every dense layer.
- **Optimizer** (`GradientDescent` / `Rprop` / `Adam`): chosen via `optimizerType`. Each one keeps its
  own state between steps (Adam: `m`, `v`, `t`; Rprop: step sizes `delta`, previous gradient `g_old`).
- **ActivationFunction** (`ReLU` / `Sigmoid` / `Identity`): one per hidden layer, from `config.activations`
  via `createActivation(activationName)`.

## 2. Data layout conventions

- A weight matrix between layer *A* (M neurons) and layer *A+1* (N neurons) has shape **(M+1) × N**.
  **Row 0 holds the thresholds θ** (bias) of the N neurons; rows 1…M are the real weights.
- To make that work, every layer input is **extended with a constant −1 at index 0**
  (`neuronValuesExt`), so `net_j = Σ_i w_ij · o_i` with `w_0j = θ_j`, `o_0 = −1`.
- One weight matrix per transition between layers. `m`, `v` (Adam) and `g` have the same shapes as the weights.
- Initialization: weights ~ N(0, √(2/M)) (He init) for rows 1…M, thresholds (row 0) = 0, `m` and `v` = 0.
  Each dense layer draws from a `weightRng` derived from `config.seed`, so runs are reproducible.
- Labels are one-hot encoded by `DataPrep._oneHot()` (`classCount` columns), so the target matches the shape of
  the softmax output.

## 3. Workflow of a training run

```mermaid
flowchart TD
    A["readEnvFile() → Config"]
    A --> B["createDataPrep(dataSetType)<br/>loadMnist() → DataPrep: scale, one-hot, split off validation data"]
    B --> V["validateConfig(config, dataPrep)"]
    V --> C["createCalculation(calculationType)<br/>createLayers(topology, activations, calculation, seed)<br/>→ MLP(inputLayer, hiddenLayers, outputLayer)"]
    C --> O["createOptimizer(optimizerType)"]
    O --> T["train()"]
    T --> E{"next epoch?"}
    E -- yes --> F["DataPrep.getMiniBatches(batchSize)<br/>shuffle → (imageBatch, labelBatch) pairs"]
    F --> G{"next mini-batch?"}
    G -- yes --> H["trainStep(imageBatch, labelBatch)"]
    H --> H1["MLP.forwardPropagation(batch) → prediction"]
    H1 --> I["MLP.calculateLoss(prediction, target) → CCE"]
    I --> J["MLP.backwardPropagation(target) → gradients"]
    J --> K["Optimizer.optimize(g, MLP.getWeights()) → new weights"]
    K --> L["MLP.setWeights(new weights)"]
    L --> G
    G -- "no (epoch done)" --> E
    E -- no --> Z["epochErrors + trained model"]
```

### 3.1 Setup (once, in `main`)

1. **`readEnvFile()`** loads the run settings into a **`Config`**: `dataSetType`, `topology` (neurons per layer),
   `activations` (one name per hidden layer), `calculationType`, `optimizerType`, `epochCount`, `batchSize`, `seed`
   and the optimizer's step size (`learningRate` for gradient descent, `alpha` for Adam).
2. **`createDataPrep(dataSetType)`** loads the raw data (`loadMnist()` → `rawImages`, `rawLabels`) and constructs
   **`DataPrep(rawImages, rawLabels, classCount, scaleFactor, validationSize, seed)`**. Its `__init__` scales the
   pixels (`scaleFactor`, e.g. 1/255 → range 0…1), one-hot encodes the labels (`_oneHot()`) and splits off
   `validationSize` samples as validation data.
3. **`validateConfig(config, dataPrep)`** checks that the config fits the data (e.g. topology vs. input size and
   class count, one activation per hidden layer, the optimizer's step size is set).
4. **`createCalculation(calculationType)`** picks `Loop` or `Matrix`.
5. **`createLayers(topology, activations, calculation, seed)`** builds `InputLayer`, the `HiddenLayer`s (each with
   its `ActivationFunction` from `createActivation()`) and the `OutputLayer`. Every dense layer gets
   `neuronCountPrev`, `neuronCount`, the shared `calculation` and a `weightRng`; `initializeWeights(weightRng)` sets up
   its weight matrix as described in [section 2](#2-data-layout-conventions). They are handed to
   **`MLP(inputLayer, hiddenLayers, outputLayer)`**.
6. **`createOptimizer(optimizerType)`** picks `GradientDescent`, `Rprop` or `Adam`.
7. **`train()`** runs the epochs and returns `epochErrors` (one error value per epoch).

### 3.2 Per epoch (`train()`)

- **`DataPrep.getMiniBatches(batchSize)`** shuffles the training images **and labels together** and returns a list of
  `(imageBatch, labelBatch)` pairs (batch size = full dataset for full-batch training, e.g. with Rprop).
- **`trainStep(imageBatch, labelBatch)`** runs 3.3–3.6 for each batch and returns its error value.

### 3.3 Per mini-batch: forward pass

`MLP.forwardPropagation(currentBatch)` calls `forwardStep()` layer by layer:

1. **InputLayer** passes the batch through unchanged.
2. Each **HiddenLayer** (`DenseLayer.forwardStep`):
   - prepends −1 to its input → `neuronValuesExt` (stored for backprop),
   - `calculation.calculateNet(extendedInput, weights)` → `net` (stored),
   - `activate(net)` → applies its `ActivationFunction` (ReLU / Sigmoid / identity) → output to the next layer.
3. **OutputLayer**: same `net` computation, then **`softmax(net)`** over the whole layer → `prediction` (stored).

### 3.4 Loss

`MLP.calculateLoss(prediction, targetValue)` → `OutputLayer.categoricalCrossEntropy()` → error value
(for monitoring; backprop does not need it directly).

### 3.5 Backward pass

`MLP.backwardPropagation(targetValue)` walks the dense layers **from back to front**:

1. **OutputLayer** `calculateDelta(targetValue)`: the combined derivative of softmax + CCE is simply
   **δ = o − t** (prediction minus one-hot target). This is where backprop starts.
2. Every dense layer then runs **`backwardStep()`**:
   - gradient: `calculation.calculateGradient(extendedInput, δ)` = extended input × δ (includes the θ row),
   - error for the layer before: `calculation.calculateBackpropError(δ, weights without row 0)`
     → `deltaPrevLayer` (θ row is dropped because the constant −1 has no predecessor).
3. **HiddenLayer** `calculateDelta(backpropError)`: δ = backpropError × `activationFuncDerivative(net)`.
4. It stops at the first hidden layer; the InputLayer computes nothing.
5. Result: one gradient matrix per weight matrix → `List gradients`.

### 3.6 Weight update

1. `Optimizer.optimize(gradients, MLP.getWeights())` computes the new weights:
   - **GradientDescent**: `w ← w − learningRate · g`
   - **Rprop**: per-weight step size, grows by `n_pos = 1.2` if the gradient sign stays the same, shrinks by
     `n_neg = 0.5` if it flips, clamped to `[deltaMin, deltaMax]`, start `delta_null`; only the sign of `g` is used.
   - **Adam**: increments step counter `t`, updates moments `m`, `v`, bias-corrects them, `w ← w − α · m̂ / (√v̂ + ε)`.
2. `MLP.setWeights(newWeights)` writes them back into the layers.
3. Next mini-batch.

## 4. Open points (from the current diagram)

Things to settle while implementing:

- [ ] `OutputLayer.calculateDelta(targetValue)` and the abstract `calculateDelta(backpropError)` take different
      inputs. Decide whether the signatures stay different or both take one generic argument.
- [ ] `Layer` only declares `forwardStep`; `backwardStep` is defined on `DenseLayer` only (fine, since InputLayer
      has no backward step, but MLP must then only call it on dense layers).
- [ ] No evaluation step yet: `DataPrep.getValidationData()` exists, but nothing in the Framework computes
      validation/test accuracy per epoch.
- [ ] `Config` only has `learningRate` and `alpha`; the other optimizer hyperparameters (`beta_1`, `beta_2`,
      `epsilon`, Rprop's `n_pos`, …) always use their defaults. Fine for now, add fields if we want to tune them.

## Updating the diagram

1. Edit `mlp-architecture.excalidraw` on excalidraw.com, then save it back into this folder under the **same name**
   (☰ → Save to… / download, then replace the file).
2. Re-export the SVG: on excalidraw.com, ☰ → Export image → SVG, with background on, and save it as
   `mlp-architecture.svg` (overwrite). (Or ask Claude to regenerate it from the `.excalidraw` file.)
3. Update this file if the workflow changed, then commit all three together.
