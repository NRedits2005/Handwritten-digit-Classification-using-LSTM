
# MNIST Digit Classification using LSTM (TensorFlow / Keras)

This project demonstrates how a **Long Short-Term Memory (LSTM)** neural network can be applied to the **MNIST handwritten digit dataset** using **TensorFlow and Keras**.  
Instead of treating images as purely spatial data, each image is interpreted as a **sequence of rows**, making it suitable for sequence models like LSTM.

---

## 📌 Project Overview

- Dataset: MNIST (28×28 grayscale handwritten digits)
- Model Type: LSTM-based Neural Network
- Framework: TensorFlow (Keras API)
- Task: Multi-class classification (digits 0–9)

Each image is treated as a sequence of **28 timesteps**, where each timestep has **28 features**.

---

## 📂 Dataset Details

- Training samples: 60,000
- Test samples: 10,000
- Image size: 28 × 28
- Classes: 10 (digits 0 to 9)

The dataset is automatically downloaded using:
```python
from tensorflow.keras.datasets import mnist
```

---

## ⚙️ Data Preprocessing

- Images are reshaped to `(samples, 28, 28)`
- Pixel values are normalized to the range **[0, 1]**
- Labels are converted to **one-hot encoded vectors** using `to_categorical`

---

## 🧠 Model Architecture

```
Input Layer      : (28 timesteps, 28 features)
LSTM Layer       : 128 units
Dense Output     : 10 units with Softmax activation
```

- Loss Function: Categorical Crossentropy  
- Optimizer: RMSprop  
- Evaluation Metric: Accuracy  

---

## 🚀 Model Training

- Epochs: 10  
- Batch Size: 128  
- Validation Split: 10% of training data  

Training and validation accuracy and loss are tracked and visualized using Matplotlib.

---

## 📊 Performance Evaluation

The model is evaluated on the test dataset using:
```python
model.evaluate(x_test, to_categorical(y_test))
```

Outputs:
- Test Accuracy
- Test Loss

---

## 🔍 Prediction Visualization

A sample test image is selected and displayed along with:
- True label (`y_test`)
- Predicted probability distribution (`y_pred`)

This helps visually inspect model predictions.

---

## 📈 Visualizations Included

- Training Accuracy vs Validation Accuracy
- Training Loss vs Validation Loss
- Sample test image with prediction

---

## 🛠️ Requirements

Install the required dependencies using:

```bash
pip install tensorflow matplotlib numpy
```

---

## 📌 Notes

- LSTMs are typically used for sequence data; using them on images is experimental and educational.
- CNNs usually outperform LSTMs on image-based tasks like MNIST.

---

## ✅ Conclusion

This project shows how sequence models like LSTMs can be adapted for image classification by reinterpreting images as sequences.  
It’s a great experiment to understand the flexibility of neural network architectures.

---

Happy Learning 🚀
