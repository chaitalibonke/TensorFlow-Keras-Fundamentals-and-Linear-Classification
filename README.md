# TensorFlow and Keras: Fundamentals and Linear Classification

## Overview
This project provides a hands-on practical implementation of core TensorFlow
and Keras concepts, including tensor operations, automatic differentiation
using GradientTape, building custom layers, and training a linear classifier
on synthetic 2D data. It serves as a foundational deep learning reference
built entirely from scratch using low-level TensorFlow operations.

---

## Objective
- Understand and manipulate TensorFlow tensors and variables
- Apply mathematical operations using TensorFlow built-in functions
- Use the GradientTape API for automatic differentiation
- Compute second-order gradients using nested GradientTape
- Build and train a linear classifier from scratch using gradient descent
- Implement a custom Dense layer as a Keras Layer subclass
- Train and evaluate a Sequential Keras model with validation data
- Perform inference using the predict() method

---

## Topics Covered

| Topic                   | Description                                         |
|-------------------------|-----------------------------------------------------|
| Tensor Operations       | Creating, modifying, and computing with tensors     |
| TensorFlow Variables    | Mutable state using assign, assign_add, assign_sub  |
| GradientTape API        | Automatic differentiation and second-order gradients|
| Linear Classifier       | Affine transformation trained with MSE loss         |
| Custom Keras Layer      | SimpleDense implemented as a Layer subclass         |
| Sequential Model        | Building models with stacked Dense layers           |
| Training and Validation | fit(), evaluate(), and predict() workflows          |

---

## Methodology

### 1. Tensor Basics
- Create constant tensors using tf.ones() and tf.zeros()
- Generate random tensors using tf.random.normal() and tf.random.uniform()
- Understand why tensors are immutable and cannot be directly assigned
- Use tf.Variable for mutable state with assign(), assign_add(),
  and assign_sub() methods

### 2. Mathematical Operations
- Apply element-wise operations: tf.square(), tf.sqrt()
- Perform matrix multiplication using tf.matmul()
- Chain tensor operations to build computational graphs

### 3. GradientTape API
- Use tf.GradientTape to automatically compute gradients of a loss
  with respect to trainable variables
- Watch constant tensors explicitly using tape.watch()
- Compute second-order gradients using nested GradientTape contexts
- Apply to a physics example: position → speed → acceleration

### 4. Linear Classifier from Scratch
- Generate two classes of 1000 random 2D points each using
  np.random.multivariate_normal()
- Stack into a single input array of shape (2000, 2) with
  corresponding binary targets
- Define weight matrix W and bias b as TensorFlow variables
- Implement the forward pass: prediction = W · input + b
- Define Mean Squared Error loss function
- Run 40 training steps using gradient descent with learning rate 0.1
- Loss converges and stabilizes at 0.0293 after 40 steps
- Visualize the decision boundary overlaid on the scatter plot

### 5. Custom Keras Layer
- Implement SimpleDense as a subclass of keras.layers.Layer
- Define weights in build() using add_weight()
- Define forward pass computation in call()
- Apply optional activation function within call()
- Test with input tensor of shape (2, 784)

### 6. Sequential Model with Keras
- Build a Sequential model with stacked Dense layers
- Compile with RMSprop optimizer and Mean Squared Error loss
- Use BinaryAccuracy as the evaluation metric
- Shuffle data and reserve 30% for validation to avoid class imbalance
- Train with fit() using batch size of 16 over 5 epochs
- Monitor training loss, validation loss, and binary accuracy per epoch
- Evaluate model using evaluate() on validation data
- Run inference using predict() on new inputs

---

## Training Results

### Linear Classifier — Scratch Implementation
| Step | Loss   |
|------|--------|
| 0    | 3.6921 |
| 10   | 0.0772 |
| 20   | 0.0469 |
| 30   | 0.0343 |
| 39   | 0.0293 |

> Loss stabilized at 0.0293 after 40 training steps.

### Keras Sequential Model — Validation Results
| Epoch | Train Loss | Val Loss | Val Accuracy |
|-------|------------|----------|--------------|
| 1     | 0.2973     | 0.0462   | 99.67%       |
| 2     | 0.0733     | 0.0245   | 100.00%      |
| 3     | 0.0779     | 0.0529   | 97.50%       |
| 4     | 0.0753     | 0.0932   | 97.00%       |
| 5     | 0.0710     | 0.0608   | 97.83%       |

> Final validation loss: 0.0608 | Final validation accuracy: 97.83%

---

## Key Findings
- TensorFlow tensors are immutable — use tf.Variable whenever
  mutable state is needed during training
- The GradientTape API provides flexible and precise control over
  gradient computation, including second-order gradients
- The linear classifier successfully separates two 2D Gaussian blobs
  with loss converging to 0.0293 after 40 steps
- The custom SimpleDense layer demonstrates that Keras layers are
  straightforward to implement from scratch using build() and call()
- The Sequential Keras model achieves 97.83% validation accuracy
  on the binary classification task within just 5 epochs
- Choosing the correct loss function is critical — a misaligned
  objective will cause the network to optimize for the wrong goal

---

## Tech Stack
- Python
- TensorFlow
- Keras
- NumPy
- Matplotlib

---

## How to Run

Install dependencies:
```bash
pip install tensorflow numpy matplotlib
```

Run the notebook:
```bash
jupyter notebook tensorflow_keras_fundamentals.ipynb
```

---

## Output
- Scatter plot of two randomly generated point classes
- Decision boundary visualization overlaid on classified data points
- Training loss curve across 40 gradient descent steps
- Epoch-by-epoch training and validation metrics

---

## Future Work
- Extend the linear classifier to a multi-layer neural network
- Add dropout and batch normalization layers
- Apply the custom layer implementation to a real-world dataset
- Experiment with different optimizers and learning rate schedules
- Implement early stopping to prevent overfitting

---

## Conclusion
This project demonstrates the foundational building blocks of deep learning
using TensorFlow and Keras — from low-level tensor manipulation and automatic
differentiation through to training a production-style Sequential model with
validation monitoring. These fundamentals form the backbone of all modern
deep learning workflows.
