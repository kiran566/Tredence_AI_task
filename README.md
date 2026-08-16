# Self-Pruning Neural Network

A PyTorch implementation of a self-pruning neural network using
learnable gates to identify and suppress unnecessary connections
during training.

This project was developed as part of the Tredence AI case study.

---

## 1. Problem Statement

The goal is to build a neural network that can learn which of its
connections are important and automatically suppress unnecessary
connections during training.

Instead of using a fixed pruning procedure after training, the model
introduces a learnable gate for every connection.

Each connection therefore has:

- A learnable weight
- A learnable gate score

The effective connection is controlled by the gate.

---

## 2. Approach

For a connection with weight `w` and gate score `s`, the gate value is:

    gate = sigmoid(s)

The effective weight becomes:

    effective_weight = w * gate

Therefore:

- Gate close to 1 → connection remains active
- Gate close to 0 → connection is suppressed

The gate scores are learned using backpropagation along with the
normal network weights.

---

## 3. Model Architecture

The network uses CIFAR-10 images.

Each CIFAR-10 image has:

    32 × 32 × 3 = 3072

input features.

The current architecture is:

    Input
      ↓
    3072
      ↓
    PrunableLinear
      ↓
    512
      ↓
    ReLU
      ↓
    PrunableLinear
      ↓
    256
      ↓
    ReLU
      ↓
    PrunableLinear
      ↓
    10 classes

Each `PrunableLinear` layer contains:

    Weight matrix
    Bias
    Gate-score matrix

---

## 4. Learnable Gating

The custom `PrunableLinear` layer is implemented as:

    gates = sigmoid(gate_scores)

    pruned_weights = weights * gates

    output = x @ pruned_weights.T + bias

The gate scores are `nn.Parameter` objects, so they are updated by
the optimizer during training.

---

## 5. Sparsity Regularization

The model uses the following sparsity loss:

    L_sparse = sum(gates)

The complete training objective is:

    L_total = L_classification + λ × L_sparse

where:

- `L_classification` is the CrossEntropyLoss
- `L_sparse` encourages smaller gate values
- `λ` controls the strength of sparsity regularization

A larger λ places stronger pressure on the gates to become small.

---

## 6. Pruning Criterion

After training, a connection is considered pruned when:

    gate < 1e-2

The sparsity percentage is calculated as:

    sparsity =
        number of gates below 0.01
        -------------------------------- × 100
        total number of gates

This allows us to measure how many connections have effectively
been suppressed.

---

## 7. Dataset

The experiments use the CIFAR-10 dataset.

CIFAR-10 contains:

- 10 classes
- 32 × 32 RGB images
- 50,000 training images
- 10,000 test images

The images are converted to tensors using PyTorch's `ToTensor`
transform.

---

## 8. Training

Optimizer:

    Adam

Learning rate:

    0.001

Loss:

    CrossEntropyLoss + λ × Sparsity Loss

Training was initially performed for:

    5 epochs

The experiments were performed on:

    CPU

---

## 9. Lambda Experiments

Multiple values of λ were tested to investigate the relationship
between sparsity and model accuracy.

### Results

| Lambda | Test Accuracy | Sparsity |
|-------:|--------------:|---------:|
| 0.00001 | 29.59% | 0.000293% |
| 0.0001  | 29.22% | 0.000879% |
| 0.0005  | 29.46% | 0.002696% |
| 0.001   | 31.27% | 0.004278% |
| 0.005   | 31.69% | 0.014943% |
| 0.01    | 30.64% | 0.020100% |
| 0.05    | 27.49% | 0.040199% |

---

## 10. Results Analysis

The experiments show that increasing λ generally increases the
measured sparsity.

For example:

    λ = 0.00001
    sparsity = 0.000293%

while:

    λ = 0.05
    sparsity = 0.040199%

Therefore, increasing the sparsity regularization strength pushes
more gates toward the pruning threshold.

However, accuracy does not decrease monotonically.

The highest observed test accuracy was:

    31.69%

at:

    λ = 0.005

with:

    sparsity = 0.014943%

At λ = 0.05, sparsity increased further, but test accuracy dropped
to:

    27.49%

This demonstrates the trade-off between model performance and
sparsity.

---

## 11. Candidate Model

Based on the current experiments, λ = 0.005 is a promising candidate
because it achieved the highest observed test accuracy while also
producing greater sparsity than the lower λ values.

    λ = 0.005
    Test Accuracy = 31.69%
    Sparsity = 0.014943%

This value will be investigated further using gate-distribution
analysis and hard-pruning evaluation.

---

## 12. Project Structure

    self-pruning-neural-network/
    │
    ├── data/
    │
    ├── models/
    │   ├── __init__.py
    │   ├── prunable_linear.py
    │   └── network.py
    │
    ├── training/
    │   ├── __init__.py
    │   ├── train.py
    │   └── evaluate.py
    │
    ├── experiments/
    │   ├── run_experiments.py
    │   └── analyze_best_model.py
    │
    ├── utils/
    │   ├── __init__.py
    │   ├── metrics.py
    │   └── plot_gate_distribution.py
    │
    ├── results/
    │
    ├── main.py
    ├── requirements.txt
    └── README.md

---

## 13. Running the Project

Create and activate a virtual environment:

    python -m venv aitask

Windows:

    aitask\Scripts\activate

Install dependencies:

    pip install -r requirements.txt

Run the baseline model:

    python main.py

Run lambda experiments:

    python experiments/run_experiments.py

Analyze the candidate model:

    python experiments/analyze_best_model.py

---

## 14. Requirements

The project uses:

- Python
- PyTorch
- torchvision
- matplotlib

Install them using:

    pip install torch torchvision matplotlib

---

## 15. Current Status

### Completed

- [x] Custom `PrunableLinear` layer
- [x] Learnable gate scores
- [x] Sigmoid-based gates
- [x] Weight-gate interaction
- [x] Sparsity regularization
- [x] CIFAR-10 data loading
- [x] Training loop
- [x] Test accuracy evaluation
- [x] Sparsity calculation
- [x] Multi-λ experiments
- [x] Accuracy-sparsity analysis

### In Progress

- [ ] Gate distribution visualization
- [ ] Hard pruning
- [ ] Before/after pruning comparison
- [ ] Final model evaluation
- [ ] Final experiment plots

---

## 16. Future Improvements

Possible improvements include:

- More efficient network architectures
- Longer training schedules
- Better initialization of gate scores
- Alternative sparsity regularizers
- Structured pruning
- Layer-wise pruning analysis
- Measuring inference speed and parameter reduction
- Comparing against standard pruning approaches

---

## 17. Conclusion

This project demonstrates a learnable self-pruning mechanism in which
each neural-network connection is controlled by a trainable gate.

The experiments show that the sparsity regularization coefficient λ
has a direct effect on the number of connections pushed below the
pruning threshold.

The current experiments identify λ = 0.005 as a promising candidate,
achieving 31.69% test accuracy and 0.014943% measured sparsity.

Further hard-pruning and gate-distribution analysis will be used to
determine the final effectiveness of the approach.