import re
import numpy as np
import pandas as pd
import math
from time import time
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from tensorflow.keras.preprocessing.sequence import pad_sequences as pad
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
import optuna

from models import FullyConnected, RNN, RNNEmbed, LSTM, BiLSTM, Transformer

class PfamDataset(Dataset):
    ''' Class for the train, validation and test datasets '''
    
    # Initiating the label encoder
    label_encoder = LabelEncoder()
    
    # Mapping
    amino_acids = ["A","C","D","E","F","G","H","I","K","L",
                   "M","N","P","Q","R","S","T","V","W","Y","X"]
    mapping = {aa:i + 1 for i, aa in enumerate(amino_acids)}
    mapping.update({'X': 21, 'U': 21, 'B': 21, 'O': 21, 'Z': 21})
    
    def __init__(self, 
                 title: str, 
                 num_classes: int, 
                 k: int, 
                 max_seq_len: int, 
                 oversampling: bool) -> None:
        '''
        Args:
            title: title/name of the dataset
            num_classes: number of classes in the dataset
            k: "skip" amount in classes (by "popularity") when creating the data
            max_seq_len: maximum length of the input sequences
            oversampling: method of oversampling used 
            
        Returns:
            None            
        '''
        self.title = title
        self.k = k
        self.num_classes = num_classes
        self.oversampling = oversampling
        self.max_seq_len = max_seq_len
        self.data = None
        self.X = None
        self.y = None
        self.weights = None
        self.len = None
        self.__initiate__()
        
    def __len__(self):
        ''' Returns the length of the dataset '''
        return len(self.X)

    def __getitem__(self, idx):
        ''' Returns a single input and label given the index input '''
        X = self.X[idx]
        y = self.y[idx]
        return X, y
    
    def __initiate__(self):
        ''' Loads the datasets and generates X and  Y'''
        path_post = "" if self.k == "all" else "_" + str(self.k)
        self.data = pd.read_csv("data/" + self.title + path_post + ".csv")
        self.X = self.get_inputs()
        self.y = self.get_labels()
        self.weights = self.get_class_weights()
        
    def replace_uncommon_amino_acids(self):
        ''' Replaces uncommon amino acids in sequences with a collective "X" '''
        replace_minority_aa = lambda data: [re.sub(r"[XUBOZ]", "X", sequence) \
                                            for sequence in data["sequence"]]
        self.data["sequence"] = replace_minority_aa(self.data)
        
    def get_inputs(self, padding="pre"):
        ''' Returns the sequences in a format suitable for our model '''
        self.replace_uncommon_amino_acids()
        encode = lambda sequence: [PfamDataset.mapping[aa] for aa in sequence[:self.max_seq_len - 2]]
        sequence_list = [encode(sequence) for sequence in self.data["sequence"]]
        sequence_list = [[22] + sequence + [23] for sequence in sequence_list]
        return torch.tensor(pad(sequence_list, maxlen=self.max_seq_len, 
                                padding=padding, truncating="post")).float()
        
    def get_labels(self):
        ''' Return the labels in a format suitable for the model'''
        if self.title == "train" or self.title == "test":
            encoded_labels = self.label_encoder.fit_transform(self.data["family_accession"])
        else:
            encoded_labels = self.label_encoder.transform(self.data["family_accession"])
        return F.one_hot(torch.tensor(encoded_labels).long(), 
                         num_classes=self.num_classes).float()
    
    def get_class_weights(self):
        ''' Returns the calss weights for use in WeightedRandomSampler'''
        class_counts = self.data["family_accession"].value_counts()
        class_counts_dict = (1 / class_counts).to_dict()
        if self.oversampling == "regular":
            self.data["family_weight"] = [class_counts_dict[x] for x in self.data["family_accession"]]
        elif self.oversampling == "sqrt":
            self.data["family_weight"] = [np.sqrt(class_counts_dict[x]) for x in self.data["family_accession"]]
        elif self.oversampling == "beta":
            beta = 0.9
            class_counts_dict = class_counts.to_dict()
            class_counts_dict = {i: 1 / ((1 - beta**c)/(1 - beta)) for (i, c) in class_counts_dict.items()}
            self.data["family_weight"] = [class_counts_dict[x] for x in self.data["family_accession"]]
        else:
            self.data["family_weight"] = [1] * len(self.data["family_accession"])
        self.len = len(self.data["family_weight"])
        return self.data["family_weight"].to_list()

def train(model, dataloader, loss_function, optimizer, device):
    ''' Training loop for model of single epoch
    
    Args:
        model: pytorch model for prediciton and training
        dataloader: pytorch dataloader for generating the model input
        loss_function: loss function for training
        optimizer: optimizer used for pytorch backpropagation
        device: device the model is ran on
        
    Returns:
        train_loss: training loss from across the epoch
        train_f1: f1 score from across the epoch
    '''
    model.train()

    train_loss = 0.0
    all_y = []
    all_preds = []

    for X, y in dataloader:
        X = X.to(device)
        y = y.to(device)
            
        y_pred = model(X)
            
        all_y.append(y)
        all_preds.append(y_pred)

        optimizer.zero_grad()
            
        loss = loss_function(y_pred, y)
        
        loss.backward()
        optimizer.step()

        train_loss += loss.data.item()
            
    # Training loss
    train_loss = train_loss / len(dataloader.dataset)
    
    # Calculate f1 score
    all_preds = torch.cat(all_preds, dim=0).cpu().argmax(dim=1)
    all_y = torch.cat(all_y, dim=0).cpu().argmax(dim=1)
    train_f1 = f1_score(all_y, all_preds, average='macro')
    
    return train_loss, train_f1

def evaluate(model, dataloader, loss_function, device):
    ''' Training loop for model of single epoch
    
    Args:
        model: pytorch model for prediciton and training
        dataloader: pytorch dataloader for generating the model input
        loss_function: loss function for training
        device: device the model is ran on
        
    Returns:
        train_loss: training loss from across the epoch
        train_f1: f1 score from across the epoch
    '''
    model.eval()

    val_loss = 0.0
    all_y = []
    all_preds = []

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)

            y_pred = model(X)

            all_y.append(y)
            all_preds.append(y_pred)

            loss = loss_function(y_pred, y)

            val_loss += loss.data.item()
            
    # Training loss
    val_loss = val_loss / len(dataloader.dataset)
    
    # Calculate f1 score
    all_preds = torch.cat(all_preds, dim=0).cpu().argmax(dim=1)
    all_y = torch.cat(all_y, dim=0).cpu().argmax(dim=1)
    val_f1 = f1_score(all_y, all_preds, average='macro')
    
    return val_loss, val_f1

def train_evaluate_loop(
    model,
    device,
    label: str,
    max_seq_len: int,
    vocab_len: int,
    batch_size: int,
    num_classes: int,
    class_gap: int,
    oversampling: str = "none",
    epochs: int = 5,
    lr: float = 0.001,
    logging: bool = True):
    ''' The main loop for training the models 
    
    Args:
        model: pytorch model for training and validating
        device: device for pytorch model training and validation
        label: name used for tensorboard summary and model saving
        max_seq_len: maximum length of the input sequences
        vocab_len: length of the vocabulary used
        num_classes: number of classes in the dataset
        class_gap: (aka k value) "skip" amount in classes 
            (by "popularity") when creating the data
        batch_size: batch size used in models
        model parameters: model type and hyperparameters
        oversampling: method of oversampling used
        epochs: number of training epochs
        lr: learning rate of the model
        
    Return:
        best_model_val_f1: validation f1 score from best model
        best_model: model with the best validation f1
    '''
    
    
    # Tensorboard writer and logging
    if logging:
        writer = SummaryWriter(log_dir="runs/" + label)
    
    # Creating the datasets
    train_dataset = PfamDataset("train", num_classes, class_gap, 
                                max_seq_len, oversampling)
    validation_dataset = PfamDataset("validation", num_classes, class_gap, 
                                     max_seq_len, oversampling="regular")
    
    # Weighted Random Sampler
    train_sampler = WeightedRandomSampler(train_dataset.weights, train_dataset.len)
    validation_sampler = WeightedRandomSampler(validation_dataset.weights, validation_dataset.len)
    
    # Creating the dataloaders
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, 
                                                   num_workers=0, sampler=train_sampler)
    validation_dataloader = torch.utils.data.DataLoader(validation_dataset,
                                                        batch_size=batch_size, 
                                                        num_workers=0, sampler=validation_sampler)
    
    # Loss function and optimizer
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    
    best_model_val_f1 = 0.0

    # Epoch loop
    for epoch in range(1, epochs+1):
        # Timing
        start = time()
        
        # Training
        train_loss, train_f1 = train(model, train_dataloader, loss_function, optimizer, device)

        # Validation
        val_loss, val_f1 = evaluate(model, validation_dataloader, loss_function, device)

        if epoch == 1 or epoch % 5 == 0:
          print('Epoch %3d/%3d, train loss: %3.2f, train f1: %3.2f, val loss: %3.2f, val f1: %3.2f, duration: %3.1fs'% \
                (epoch, epochs, train_loss, train_f1, val_loss, val_f1, time() - start))

        if logging:
            writer.add_scalar("Loss/Train", train_loss, epoch)
            writer.add_scalar("F1/Train", train_f1, epoch)
            writer.add_scalar("Loss/Eval", val_loss, epoch)
            writer.add_scalar("F1/Eval", val_f1, epoch)
        
        if val_f1 > best_model_val_f1:
            best_model = deepcopy(model)
            best_model_val_f1 = val_f1
    
    if logging:
        writer.close()
    
    return best_model_val_f1, best_model

def hyperparameter_tuning(
    trial, 
    device,
    model_name: str, 
    model_parameters: dict, 
    max_seq_len: int, 
    num_classes: int, 
    class_gap: int,
    vocab_len: int, 
    batch_size: int, 
    epochs: int = 5):
    ''' Returns the validation f1 of a training cycle with tested hyperparaters 
    
    Args:
        trial: trial information from optuna hyperparameter tuning
        device: device for pytorch model training and validation
        model_name: type of model that is being optimised in trial i.e. BiLSTM
        max_seq_len: maximum length of the input sequences
        num_classes: number of classes in the dataset
        num_classes: number of classes in the dataset
        class_gap: (aka k value) "skip" amount in classes 
        vocab_len: length of the vocabulary used
        batch_size: batch size used in models
        epochs: number of epochs for training
    
    Returns:
        val_f1: validation f1 score from this trial
    '''
    
    # Model type selection
    if model_name == "fc":
        # Optuna selecting parameters according to a Tree-structured 
        # Parzen Estimator (TPE), which is a Bayesian optimization algorithm
        hidden_dim = trial.suggest_int("hidden_dim", 64, 512)
        num_layers = trial.suggest_int("num_layers", 0, 2)
        # Structuring the model
        model = FullyConnected(
            dropout = 0.2,
            input_dim = max_seq_len,
            hidden_dim = hidden_dim,
            output_dim = num_classes,
            num_layers = num_layers,
        )
        print(f"FC Trial {trial.number} | hd {hidden_dim}, nl {num_layers}")
    elif model_name == "rnn":
        hidden_dim = trial.suggest_int("hidden_dim", 64, 256)
        num_layers = trial.suggest_int("num_layers", 2, 4)
        # Restraining the model size
        if hidden_dim * num_layers > 1200:
            return None
        model = RNN(
            dropout = 0.2,
            hidden_dim = hidden_dim, 
            num_layers = num_layers,
            vocab_len = vocab_len,
            output_dim = num_classes, 
            device = device, 
        )
        print(f"RNN Trial {trial.number} | hd {hidden_dim}, nl {num_layers}")
    elif model_name == "rnn_embed":
        hidden_dim = trial.suggest_int("hidden_dim", 64, 512)
        num_layers = trial.suggest_int("num_layers", 2, 4)
        embed_size = trial.suggest_int("embed_size", 16, 64)
        # Restraining the model size
        if hidden_dim * num_layers > 1200:
            return None
        model = RNNEmbed(
            dropout = 0.2,
            hidden_dim = hidden_dim, 
            num_layers = num_layers,
            embed_size = embed_size,
            vocab_len = vocab_len,
            output_dim = num_classes, 
            device = device, 
        )
        print(f"RNNEmbed Trial {trial.number} | hd {hidden_dim}, nl {num_layers}, ne {embed_size}")
    elif model_name == "lstm":
        hidden_dim = trial.suggest_int("hidden_dim", 64, 256)
        num_layers = trial.suggest_int("num_layers", 2, 4)
        embed_size = trial.suggest_int("embed_size", 16, 64)
        # Restraining the model size
        if hidden_dim * num_layers > 900:
            return None
        model = LSTM(
            dropout = 0.2,
            hidden_dim = hidden_dim, 
            num_layers = num_layers,
            embed_size = embed_size,
            vocab_len = vocab_len,
            output_dim = num_classes, 
            device = device, 
        )
        print(f"LSTM Trial {trial.number} | hd {hidden_dim}, nl {num_layers}, ne {embed_size}")
    elif model_name == "bi-lstm":
        hidden_dim = trial.suggest_int("hidden_dim", 64, 256)
        num_layers = trial.suggest_int("num_layers", 2, 4)
        embed_size = trial.suggest_int("embed_size", 16, 64)
        # Restraining the model size
        if hidden_dim * num_layers > 1000:
            return None
        model = BiLSTM(
            dropout = 0.2,
            hidden_dim = hidden_dim, 
            num_layers = num_layers,
            embed_size = embed_size,
            vocab_len = vocab_len,
            output_dim = num_classes, 
            device = device, 
        )
        print(f"BiLSTM Trial {trial.number} | hd {hidden_dim}, nl {num_layers}, ne {embed_size}")
    elif model_name == "transformer":
        embed_size = trial.suggest_categorical("embed_size", [16, 32, 64, 128, 256, 512])
        hidden_dim = trial.suggest_int("hidden_dim", max(128, embed_size), 512)
        feed_forward_dim = trial.suggest_int("feed_forward_dim", 64, 512)
        num_layers = trial.suggest_int("num_layers", 2, 4)
        num_heads = trial.suggest_categorical("num_heads", [4, 8, 16])
        # Restraining the model size
        if hidden_dim * num_layers > 1024:
            return None
        model = Transformer(
            dropout = 0.2,
            hidden_dim = hidden_dim, 
            feed_forward_dim = feed_forward_dim, 
            embed_size = embed_size,
            num_layers = num_layers,
            num_heads = num_heads,
            vocab_len = vocab_len,
            output_dim = num_classes,
        )
        print(f"Transformer Trial {trial.number} | hd {hidden_dim}, nl {num_layers}, ne {embed_size}, ff {feed_forward_dim}, nh {num_heads}")
        
    # Label
    label = "optimize_" + model_name + "_trial_" + str(trial.number)
    
    # Model device allocation
    model = model.to(device)
    
    # Training and validating the model
    best_model_val_f1, _ = train_evaluate_loop(
        model = model, 
        device = device, 
        label = label, 
        epochs = epochs,
        max_seq_len = max_seq_len,
        vocab_len = vocab_len,
        batch_size = batch_size,
        num_classes = num_classes,
        class_gap = class_gap,
        logging=False
    )
    
    return best_model_val_f1

def training_with_parameters(
    device,
    model_parameters: dict, 
    model_name: str, 
    max_seq_len: int, 
    num_classes: int, 
    class_gap: int,
    vocab_len: int, 
    batch_size: int, 
    epochs: int, 
    oversampling: str = "NA") -> float:
    ''' Returns the validation f1 of a training cycle with tested hyperparaters 
    
    Args:
        device: device for pytorch model training and validation
        model_parameters: dictionary of hyperparameters for model
        model_name: type of model that is being optimised in trial i.e. BiLSTM
        num_classes: number of classes in the dataset
        class_gap: (aka k value) "skip" amount in classes 
        max_seq_len: maximum length of the input sequences
        vocab_len: length of the vocabulary used
        batch_size: batch size used in models
        epochs: number of epochs for training
    
    Returns:
        val_f1: validation f1 score from this trial
    '''
    
    # Model type selection
    if model_name == "fc":
        # Creating the model with best parameters
        model = FullyConnected(
            dropout = 0.2,
            hidden_dim = model_parameters["hidden_dim"], 
            num_layers = model_parameters["num_layers"],
            input_dim = max_seq_len, 
            output_dim = num_classes, 
        )
        print(f'FC Model Best Params | hd {model_parameters["hidden_dim"]}, nl {model_parameters["num_layers"]}')
    elif model_name == "rnn":
        model = RNN(
            dropout = 0.2,
            hidden_dim = model_parameters["hidden_dim"], 
            num_layers = model_parameters["num_layers"],
            vocab_len = vocab_len,
            output_dim = num_classes, 
            device = device, 
        )
        print(f'RNN Model Best Params  | hd {model_parameters["hidden_dim"]}, nl {model_parameters["num_layers"]}')
    elif model_name == "rnn_embed":
        model = RNNEmbed(
            dropout = 0.2,
            hidden_dim = model_parameters["hidden_dim"], 
            num_layers = model_parameters["num_layers"],
            embed_size = model_parameters["embed_size"],
            vocab_len = vocab_len,
            output_dim = num_classes, 
            device = device, 
        )
        print(f'RNNEmbed Model Best Params  |  hd {model_parameters["hidden_dim"]}, nl {model_parameters["num_layers"]}, ne {model_parameters["embed_size"]}')
    elif model_name == "lstm":
        model = LSTM(
            dropout = 0.2,
            hidden_dim = model_parameters["hidden_dim"], 
            num_layers = model_parameters["num_layers"],
            embed_size = model_parameters["embed_size"],
            vocab_len = vocab_len,
            output_dim = num_classes, 
            device = device, 
        )
        print(f'LSTM Model Best Params  | hd {model_parameters["hidden_dim"]}, nl {model_parameters["num_layers"]}, ne {model_parameters["embed_size"]}')
    elif model_name == "bi-lstm":
        model = BiLSTM(
            dropout = 0.2,
            hidden_dim = model_parameters["hidden_dim"], 
            num_layers = model_parameters["num_layers"],
            embed_size = model_parameters["embed_size"],
            vocab_len = vocab_len,
            output_dim = num_classes, 
            device = device, 
        )
        print(f'BiLSTM Model Best Params  | hd {model_parameters["hidden_dim"]}, nl {model_parameters["num_layers"]}, ne {model_parameters["embed_size"]}')
    elif model_name == "transformer":
        model = Transformer(
            dropout = 0.2,
            hidden_dim = model_parameters["hidden_dim"], 
            num_layers = model_parameters["num_layers"],
            feed_forward_dim = model_parameters["feed_forward_dim"], 
            num_heads = model_parameters["num_heads"],
            embed_size = model_parameters["embed_size"],
            vocab_len = vocab_len,
            output_dim = num_classes,
        )
        print(f'Transformer Model Best Params  | hd {model_parameters["hidden_dim"]}, nl {model_parameters["num_layers"]}, ne {model_parameters["embed_size"]}, ff {model_parameters["feed_forward_dim"]}, nh {model_parameters["num_heads"]}')
        
    # Label
    if oversampling == "NA":
        label = model_name + "_iter" + str(class_gap)
        oversampling = "none"
    else:
        label = model_name + "_iter" + str(class_gap) + "_oversampling_" + oversampling
    
    # Model device allocation
    model = model.to(device)
    
    # Training and validating the model
    best_model_val_f1, best_model = train_evaluate_loop(
        model = model, 
        device = device, 
        label = label, 
        epochs = epochs,
        max_seq_len = max_seq_len,
        vocab_len = vocab_len,
        batch_size = batch_size,
        num_classes = num_classes,
        class_gap = class_gap,
        oversampling = oversampling,
    )
    
    # Saving the best model
    torch.save(best_model, "model/" + label + ".pt")
    
    return best_model_val_f1

def tuning_and_training(
    model_name: str, 
    no_trials: int,
    num_classes: int, 
    class_gap: int,
    vocab_len: int, 
    max_seq_len: int, 
    batch_size: int,
    model_parameters: dict = {}, 
    epochs: int = 10) -> None:
    ''' Uses Optuna to fine-tune hyperparameters then trains and saves best model 
    
    Args:
        model_name: type of model that is being optimised in trial i.e. BiLSTM
        no_trials: number of trials in the optuna fine-tuning
        num_classes: number of classes in the dataset
        class_gap: (aka k value) "skip" amount in classes 
        max_seq_len: maximum length of the input sequences
        vocab_len: length of the vocabulary used
        batch_size: batch size used in models
        epochs: number of epochs for training
        
    Returns:
        None
    '''
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Hyperparameter fine-tuning
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: hyperparameter_tuning(
            trial,
            device = device,
            model_name = model_name,
            model_parameters = model_parameters,
            max_seq_len = max_seq_len,
            num_classes = num_classes,
            class_gap = class_gap,
            vocab_len = vocab_len,
            batch_size = batch_size,
        ),
        n_trials = no_trials,
    )
    model_parameters = study.best_params
    print("Best parameters:", model_parameters)
    
    # Training model with best hyperparameters
    val_f1 = training_with_parameters(
        device = device, 
        model_parameters = model_parameters, 
        model_name = model_name, 
        num_classes = num_classes,
        class_gap = class_gap,
        max_seq_len = max_seq_len, 
        vocab_len = vocab_len, 
        batch_size = batch_size,
        epochs = epochs,
    )

def test(
    label: str,
    oversampling: str = "none",
    num_classes: int = 100,
    class_gap: int = 1,
    max_seq_len: int = 128,
    vocab_len = 24,
    batch_size: int = 64):
    ''' Evaluates the models against the test set returning the loss and f1 score
    
    Args:
        device: device for pytorch model training and validation
        label: name used to load model
        oversampling: method of oversampling used
        num_classes: number of classes in the dataset
        class_gap: (aka k value) "skip" amount in classes 
            (by "popularity") when creating the data
        max_seq_len: maximum length of the input sequences
        vocab_len: length of the vocabulary used
        batch_size: size of the batch used in training and evaluation
        
    Return:
        test_loss: loss according to the loss function
        test_f1: f1 score of the predicted values
    '''
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Creating the datasets
    test_dataset = PfamDataset("test", num_classes, class_gap, max_seq_len, oversampling)
    
    # Weighted Random Sampler
    sampler = WeightedRandomSampler(test_dataset.weights, test_dataset.len)
    
    # Creating the dataloaders
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=batch_size, 
                                                  num_workers=0, sampler=sampler)
    
    # Model
    model = torch.load("model/" + label + ".pt")
    model = model.to(device)
    
    # Loss function
    loss_function = nn.CrossEntropyLoss()
    
    # Testing
    test_loss, test_f1 = evaluate(model, test_dataloader, loss_function, device)
    print(f"Test F1 score: {test_f1}")

    return test_loss, test_f1

def oversampling_training(
    methods: list[str], 
    class_gaps: list[int], 
    model_parameters: dict,
    num_classes: int = 100, 
    max_seq_len: int = 128, 
    vocab_len: int = 24, 
    batch_size: int = 64,
    epochs: int = 10):
    ''' Trains and evaluates a model with varying oversampling methods
    and varying distributions within the datasets.
    
    Args:
        methods: list of oversampling methods used
        class_gaps: "skips" used when selecting classes to be 
            included in the dataset once ordered
        model_parameters: parameters in transformer model
        num_classes: number of classes used in dataset
        max_seq_len: maximum length of the input sequences
        vocab_len: length of the vocabulary used
        batch_size: size of the batch used in training and evaluation
        epochs: number of training epochs

    Returns:
        None
    '''

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Training loop
    for oversampling in methods:
        for class_gap in class_gaps:
            print(f"Training gap {class_gap} with oversampling {oversampling}.")

            # Training model with best hyperparameters
            val_f1 = training_with_parameters(
                device = device, 
                model_parameters = model_parameters, 
                model_name = "transformer", 
                num_classes = 100,
                class_gap = class_gap,
                max_seq_len = max_seq_len, 
                vocab_len = vocab_len, 
                batch_size = batch_size,
                epochs = epochs,
                oversampling = oversampling,
            )

            print("\n\n")

            label = "transformer_iter" + str(class_gap) + "_oversampling_" + oversampling
            _, _ = test(label, class_gap=class_gap, oversampling="regular")

            print("\n\n")