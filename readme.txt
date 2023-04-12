# Protein Family Classification using Deep Learning
This repository contains code for conducting experiments on protein family classification using deep learning. The code is based on the PFAM dataset, a popular resource for protein sequence analysis. The objective of the experiments is to investigate the effectiveness of various neural network architectures for protein family classification, and to explore the impact of oversampling strategies on addressing the challenges posed by imbalanced class distributions in the dataset.

## Installation
To run the code, please install the required packages using the following command:

```bash
pip install -r requirements.txt
```

## Files
The repository contains the following files:

models.py: This file contains the definition of different neural network models used in the experiments, including Fully-Connected Feed-Forward Network, Recurrent Neural Network (RNN), RNN with embedding, Long Short-Term Memory (LSTM), Bi-Directional LSTM, and Transformer (encoder only).

operations.py: This file contains all the functions and classes required for the experiments, including data loading, preprocessing, oversampling, hyperparameter tuning, and performance evaluation.

data_analysis.ipynb: This file contains the initial exploratory analysis of the PFAM dataset, including data visualization and summary statistics.

architecture_experiment.ipynb: This file contains the code and results of the first experiment, which investigates the effectiveness of different neural network architectures for protein family classification.

oversampling_experiment.ipynb: This file contains the code and results of the second experiment, which explores the impact of oversampling strategies on addressing the challenges posed by imbalanced class distributions in the dataset.

requirements.txt: This file contains a list of required packages for running the code.

## Usage
To reproduce the experiments, please follow the steps outlined in the Jupyter notebooks. In general, the workflow involves:

Loading and preprocessing the data using the PfamDataset in operations.py, before using the DataLoader class from pytorch.
Implementing the desired neural network architecture using the corresponding model definition in models.py.
Tuning hyperparameters using the Optuna package in operations.py.
Evaluating model performance using the F1 score on a test set.
Note that the experiments were conducted on a subset of the PFAM dataset, focusing on the 100 most popular protein families to make the analysis manageable. The code can be modified to work with different subsets or the entire dataset.

## Conclusion
The experiments provide valuable insights into the most effective deep learning methods for protein family classification and shed light on the strengths and weaknesses of various neural network architectures in the context of this challenging NLP problem. The results of the experiments may have practical implications for researchers and practitioners in the field of bioinformatics and drug discovery, enabling them to more effectively classify protein sequences and identify potential drug targets for developing new treatments.