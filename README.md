# PyTorch for Computer Vision – Learning Repository

This repository documents my hands-on journey to mastering **PyTorch**, starting from tensor basics and progressing into neural networks, datasets, and machine learning workflows. Each notebook captures code experiments, explanations, and key takeaways.

---

## 📂 Repository Structure

### 1. **Intro to Tensors**

*Notebook:* `intro_to_tensors/`

* Created PyTorch tensors and explored properties like shape, dtype, and device.
* Practiced scalars, vectors, matrices, and higher-dimensional tensors.

### 2. **Tensor Indexing**

*Notebook:* `tensor_indexing/`

* Slicing and indexing to access tensor elements.
* Fancy indexing for selecting rows/columns.

### 3. **Noise Images**

*Notebook:* `Noise_images/random_number_noise_image.ipynb`

* Generated random noise images using PyTorch tensors.
* Learned the role of randomness in initialization and augmentation.

### 4. **Tensor of Zeros & Ones**

*Notebook:* `tensor_zeros_ones/tensor_of_zeros_ones.ipynb`

* Created tensors filled with zeros and ones.
* Understood why initialization matters in deep learning.

### 5. **Tensor DataTypes**

*Notebook:* `tensor_datatypes/datatypes.ipynb`

* Explored tensor data types (`float32`, `int64`, etc.).
* Learned dtype effects on precision, memory, and performance.

### 6. **Tensor Manipulation**

*Notebook:* `tensor_manipulation/tensor_manipulation.ipynb`

* Reshaped tensors using `view`, `reshape`, `squeeze`, `unsqueeze`, `transpose`.
* Understood why shape manipulation is critical for neural networks.

### 7. **View & Reshape**

*Notebook:* `view_reshape/view_reshape_operation.ipynb`

* Focused on differences between `view()` and `reshape()`.
* Understood memory sharing vs copying.

### 8. **Tensor Stack**

*Notebook:* `tensor_stack/stack_operation.ipynb`

* Combined tensors using `torch.stack` and compared with `cat()`.
* Practiced stacking in different dimensions.

### 9. **Matrix Aggregation**

*Notebook:* `matrix_aggregation/Matrix_Aggregation.ipynb`

* Reduction ops: `sum`, `mean`, `min`, `max`.
* Learned how aggregation is used for loss functions and metrics.

### 10. **Intro to Neural Network Components**

*Notebook:* `understanding_nn/intro_nn_components.ipynb`

* Defined NNs with functional API and `nn.Sequential`.
* Explored layers, activations, forward passes, and network connections.
* Built simple networks in both styles.

### 11. **Linear Regression with PyTorch**

*Notebook:* `Linear_Regression/linear_regression_using_pytorch.ipynb`

* Implemented linear regression from scratch with tensors and `nn.Module`.
* Trained using gradient descent and `MSELoss`.
* Tested different optimizers and learning rates.

### 12. **Multiclass Classification (Iris Dataset)**

*Notebook:* `multi-class-classification/multi_class_classification.ipynb`

* Built a multiclass classification model on Iris dataset.
* Used functional API and `nn.CrossEntropyLoss`.
* Practiced evaluation and accuracy measurement.

### 13. **Custom DataLoader**

*Notebook:* `concept_custom_data_loader/custom_data_loader.ipynb`

* Built a custom Dataset class by subclassing `torch.utils.data.Dataset`.
* Implemented `__init__`, `__len__`, `__getitem__` for preprocessing.
* Wrapped in `DataLoader` for batching, shuffling, parallel loading.
* Trained model with batching and epoch updates.

**Outcome:** Designed flexible and reusable data pipelines in PyTorch.

### 14. **Custom Image Dataset Loader**

*Notebook:* `custom_image_dataset_loader/image_dataset_loader.ipynb`

* Implemented an image dataset loader using PyTorch `Dataset` and `DataLoader`.
* Loaded images from directory structure (`train/`, `test/`).
* Applied transforms (`Resize`, `ToTensor`).
* Created mappings for class names and labels.
* Visualized images using Matplotlib after loading.

**Outcome:** Gained experience in handling image datasets for classification tasks.

### 15. **CNN Training on Custom Dataset**

*Notebook:* `CNN_training/cnn_training_custom_dataset.ipynb`

* Built a custom **CNN architecture** with Conv2D, BatchNorm, ReLU, MaxPooling, and fully connected layers.
* Trained on a 3-class custom image dataset (30 epochs).
* Achieved **80.5% test accuracy**.
* Saved model weights (`cnn_model.pth`).
* Implemented an **ImageClassifier class** for inference and visualization using OpenCV.

**Outcome:** End-to-end workflow for CNN training, evaluation, saving, and real-world inference.

---

## 🛠️ Tech Stack

* Python 3
* PyTorch
* NumPy
* Matplotlib
* PIL
* Torchvision
* OpenCV

---

## 📝 Notes

* This repository is strictly for **learning purposes** — documenting step-by-step exploration.
* Next steps: move to **deep learning workflows** for image classification and convolutional neural networks.

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

## 👨‍💻 Author

**Bhupen** – “This work reflects the process of continuous learning, experimenting, making mistakes, and improving with each step.”

🔗 [LinkedIn](https://www.linkedin.com/in/bhupenparmar/) | 💻 [GitHub](https://github.com/bhupencoD3)
