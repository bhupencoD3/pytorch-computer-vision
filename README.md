# PyTorch for Computer Vision – Learning Repository

This repository documents my hands-on journey to mastering PyTorch, starting from tensor basics and progressing into neural networks and machine learning workflows. Each notebook captures code experiments, explanations, and key takeaways.

---

## Repository Structure

### 1. **Intro to Tensors**

*Notebook:* `intro_to_tensors/`

* Learned to create PyTorch tensors and explored properties like shape, dtype, and device.
* Practiced scalars, vectors, matrices, and higher-dimensional tensors.

### 2. **Tensor Indexing**

*Notebook:* `tensor_indexing/`

* Worked on slicing and indexing to access tensor elements.
* Explored fancy indexing for selecting rows/columns.

### 3. **Noise Images**

*Notebook:* `Noise_images/random_number_noise_image.ipynb`

* Generated random noise images using PyTorch tensors.
* Learned the role of randomness in initialization and data augmentation.

### 4. **Tensor of Zeros & Ones**

*Notebook:* `tensor_zeros_ones/tensor_of_zeros_ones.ipynb`

* Created tensors filled with zeros and ones.
* Understood why initialization matters in deep learning.

### 5. **Tensor DataTypes**

*Notebook:* `tensor_datatypes/datatypes.ipynb`

* Explored different tensor data types like `float32`, `int64`.
* Learned how dtype affects precision, memory, and performance.

### 6. **Tensor Manipulation**

*Notebook:* `tensor_manipulation/tensor_manipulation.ipynb`

* Practiced reshaping tensors using `view`, `reshape`, `squeeze`, `unsqueeze`, and `transpose`.
* Learned why shape manipulation is critical for neural networks.

### 7. **View & Reshape**

*Notebook:* `view_reshape/view_reshape_operation.ipynb`

* Focused on differences between `view()` and `reshape()`.
* Understood memory sharing vs copying and when to prefer one.

### 8. **Tensor Stack**

*Notebook:* `tensor_stack/stack_operation.ipynb`

* Combined multiple tensors using `torch.stack` and compared it with `cat()`.
* Practiced stacking in different dimensions for batch operations.

### 9. **Matrix Aggregation**

*Notebook:* `matrix_aggregation/Matrix_Aggregation.ipynb`

* Explored reduction operations like `sum`, `mean`, `min`, and `max`.
* Learned how aggregation is used for loss functions and evaluation metrics.

### 10. **Intro to Neural Network Components**

*Notebook:* `understanding_nn/intro_nn_components.ipynb`

* Defined neural networks using both functional API and `nn.Sequential`.
* Explored layers, activations, forward passes, and network connections.
* Practiced creating simple networks in two styles to compare flexibility and readability.

### 11. **Linear Regression with PyTorch**

*Notebook:* `Linear_Regression/linear_regression_using_pytorch.ipynb`

* Implemented linear regression from scratch using PyTorch tensors and `nn.Module`.
* Trained and evaluated a model with gradient descent and `MSELoss`.
* Experimented with optimizer choices and learning rates.

### 12. **Multiclass Classification (Iris Dataset)**

*Notebook:* `multi-class-classification/multi_class_classification.ipynb`

* Implemented a multiclass classification model on the Iris dataset.
* Used PyTorch functional API and `nn.CrossEntropyLoss`.
* Practiced training, evaluation, and accuracy measurement for multiple classes.

---

## Tech Stack

* Python 3
* PyTorch
* NumPy
* Matplotlib

---

## Notes

* This repository is strictly for **learning purposes** — each notebook documents step-by-step exploration.
* Next steps: move from simple neural networks to **deep learning workflows** for image classification and convolutional neural networks.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Author

**Bhupen** – Learning & building one frame at a time
[LinkedIn](https://www.linkedin.com/in/bhupenparmar/) | [GitHub](https://github.com/bhupencoD3)
