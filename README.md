# Human Activity Recognition (HAR) using Deep Learning

**Foundations of Data Science (FDS) 2025/2026** **Sapienza University of Rome**

##  Project Overview
This project focuses on **Human Activity Recognition (HAR)** based on time-series data collected from smartphone sensors (accelerometer and gyroscope). The goal is to classify various physical activities (e.g., Walking, Standing, Laying) using Deep Learning techniques.

The project utilizes the **UCI HAR Dataset** and implements two distinct neural network architectures to compare performance and training strategies:

1.  **CNN-LSTM**: A hybrid end-to-end model combining Convolutional Neural Networks (for feature extraction) and LSTMs (for temporal sequence modeling).
2.  **ConvAE-LSTM**: A two-stage approach involving a **Convolutional Autoencoder** (pre-trained for feature compression) followed by an **LSTM Classifier** (transfer learning).

##  Project Structure
```text
Actify/
│
├── cfg/                          # Hydra configuration files
│   ├── base.yaml                 # Base configuration
│   └── net/
│       ├── cnn_lstm.yaml         # CNN-LSTM model configuration
│       └── convae_lstm.yaml      # ConvAE + LSTM configuration
│
├── data/                         # Dataset directory 
│
├── src/
│   ├── dataset/
│   │   ├── dataset.py            # Dataset download
│   │   └── loader.py             # Data loading utilities
│   │
│   └── net/
│       ├── blocks/               # Reusable network building blocks
│       │   ├── cnn.py
│       │   ├── conv_deconv.py
│       │   └── lstm.py
│       │
│       └── models/               # Model definitions
│           ├── cnn_lstm.py        # CNN-LSTM model
│           ├── convae_lstm.py     # ConvAE + LSTM model
│           └── __init__.py
│
├── experiment.py                 # Main training & evaluation pipeline
│
├── sweep_cnn_lstm.yaml            # W&B sweep configuration (CNN-LSTM)
├── sweep_convae_lstm.yaml         # W&B sweep configuration (ConvAE-LSTM)
│
├── README.md                     # Project documentation
├── pyproject.toml                # Project dependencies and configuration
├── uv.lock                       # Dependency lock file
├── .python-version               # Python version
└── .gitignore

```

##  How to Run the Project

This project uses **uv** as the Python package manager and **Hydra** for configuration management.

1. Clone the repository and enter the project folder:
   
   `!git clone https://github.com/Catairu/Actify.git`

   `!cd Actify`

2. Install the dependencies using uv:
   
   `!uv sync`

3. Download and preprocess the UCI HAR Dataset:
   
   `!uv run python src/dataset/dataset.py`

4. Select the model to run by editing the configuration file:
   cfg/base.yaml

   Set the `net name` field to one of the following values:
   - cnn_lstm : CNN-LSTM end-to-end architecture
   - convae_lstm : Convolutional Autoencoder + LSTM   

   Example:
   name: cnn_lstm
   or
   name: convae_lstm

5. Run training and evaluation:
   
   `!uv run python src/experiment.py`

Evaluation metrics:
- Accuracy
- F1-score
- Confusion Matrix 

For further details on the methodology, experiments, and results, please refer to the project report.
