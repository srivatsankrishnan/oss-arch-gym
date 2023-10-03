import os, sys
import pickle

from absl import flags
from absl import app

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import stats
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.datasets import fetch_california_housing

import torch
import torch.nn as nn
import torch.optim as optim

# Define parameters for the training/handling of the data and model
flags.DEFINE_string('data_path', './data', 'Path to the data')
flags.DEFINE_string('model_path', './models', 'Path to the model')
flags.DEFINE_integer('seed', 123, 'Seed for the random number generator')
flags.DEFINE_float('train_size', 0.8, 'the split between train and test dataset')
flags.DEFINE_enum('preprocess', None, ['normalize', 'standardize'], 'Preprocessing method')
flags.DEFINE_enum('encode', 'one_hot', ['one_hot', 'label'], 'Encoding method')
flags.DEFINE_bool('visualize', False, 'enable visualization of the data')
flags.DEFINE_bool('train', False, 'enable training of the model')
flags.DEFINE_integer('output_index', 0, 'Index of the output to train the model on')
flags.DEFINE_bool('custom_dataset', False, 'Whether to use a custom dataset or not')

# Hyperparameters for the model
flags.DEFINE_integer('n_hidden_layers', 3, 'Number of hidden layers')
flags.DEFINE_enum('activation', 'relu', ['relu', 'tanh', 'sigmoid', 'LeakyRelu', 'ELU'], 'Activation function')
flags.DEFINE_enum('output_activation', 'sigmoid', ['softmax', 'sigmoid'], 'Output activation function')
flags.DEFINE_enum('loss', 'mse', ['mse', 'l1', 'humber'], 'Loss function')
flags.DEFINE_enum('optimizer', 'adam', ['adam', 'sgd', 'rmsprop', 'LBFGS'], 'Optimizer')
flags.DEFINE_integer('n_epochs', 300, 'Number of epochs')
flags.DEFINE_integer('batch_size', 64, 'Batch size')
flags.DEFINE_float('learning_rate', 0.001, 'Learning rate')
flags.DEFINE_bool('early_stopping', True, 'Enable early stopping')
flags.DEFINE_integer('patience', 10, 'Patience for early stopping')

FLAGS = flags.FLAGS

def preprocess_data(actions, observations, exp_path):
    observations = observations.to_frame()
    # Categorical features
    categorical_cols = list(set(actions.columns) - set(actions._get_numeric_data().columns))
    if len(categorical_cols) == 0:
        NO_CAT = True
    else:
        NO_CAT = False
    categorical_actions = actions[categorical_cols]
    
    # Numerical features
    numerical_actions = actions._get_numeric_data()
    
    encoder_path = os.path.join(exp_path, 'encoder')
    if not os.path.exists(encoder_path):
        os.makedirs(encoder_path)

    # Encode categorical features
    if FLAGS.encode == 'one_hot' and not NO_CAT:
        # One-hot encode categorical features
        enc = OneHotEncoder(handle_unknown='ignore')
        enc.fit(categorical_actions)
        # Save the encoder
        path = os.path.join(encoder_path, 'one_hot_encoder.joblib')
        pickle.dump(enc, open(path, 'wb'))
        # Transform the categorical features
        dummy_col_names = pd.get_dummies(categorical_actions).columns
        categorical_actions = pd.DataFrame(enc.transform(categorical_actions).toarray(), columns=dummy_col_names)
    elif FLAGS.encode == 'label' and not NO_CAT:
        dummy_actions = pd.DataFrame()
        for categorical_col in categorical_cols:
            # Label encode categorical features
            enc = LabelEncoder()
            enc.fit(categorical_actions[categorical_col])
            # Save the encoder
            path = os.path.join(encoder_path, 'label_encoder_{}.joblib'.format(categorical_col))
            pickle.dump(enc, open(path, 'wb'))
            # Transform the categorical features
            dummy_actions[categorical_col] = enc.transform(categorical_actions[categorical_col])
        categorical_actions = pd.DataFrame(dummy_actions, columns=categorical_cols)
    elif not NO_CAT:
        raise ValueError('Encoding method not supported')

    preprocess_data_path = os.path.join(exp_path, 'preprocess_data')
    if not os.path.exists(preprocess_data_path):
        os.makedirs(preprocess_data_path)

    # Normalize numerical features
    if FLAGS.preprocess is None:
        pass
    elif FLAGS.preprocess == 'normalize':
        # Normalize numerical features for actions
        normalize_feature_transformer = MinMaxScaler(feature_range=(0, 1))
        normalized_numerical_features = normalize_feature_transformer.fit_transform(numerical_actions)
        numerical_actions = pd.DataFrame(normalized_numerical_features, columns=[numerical_actions.columns])
        # Save the scaler
        path = os.path.join(preprocess_data_path, 'normalize_feature_transformer_actions.joblib')
        pickle.dump(normalize_feature_transformer, open(path, 'wb'))
        
        # Normalize numerical features for observations
        normalize_feature_transformer = MinMaxScaler(feature_range=(0, 1))
        normalized_numerical_features = normalize_feature_transformer.fit_transform(observations)
        observations = pd.DataFrame(normalized_numerical_features, columns=[observations.columns])
        # Save the scaler
        path = os.path.join(preprocess_data_path, 'normalize_feature_transformer_observations_{}.joblib'.format(FLAGS.output_index))
        pickle.dump(normalize_feature_transformer, open(path, 'wb'))
    elif FLAGS.preprocess == 'standardize':
        # Standardize numerical features for actions
        standardize_feature_transformer = StandardScaler()
        standardized_numerical_features = standardize_feature_transformer.fit_transform(numerical_actions)
        numerical_actions = pd.DataFrame(standardized_numerical_features, columns=[numerical_actions.columns])
        # Save the scaler
        path = os.path.join(preprocess_data_path, 'standardize_feature_transformer_actions.joblib')
        pickle.dump(standardize_feature_transformer, open(path, 'wb'))
        
        # Standardize numerical features for observations
        standardize_feature_transformer = StandardScaler()
        standardized_numerical_features = standardize_feature_transformer.fit_transform(observations)
        observations = pd.DataFrame(standardized_numerical_features, columns=[observations.columns])
        # Save the scaler
        path = os.path.join(preprocess_data_path, 'standardize_feature_transformer_observations_{}.joblib'.format(FLAGS.output_index))
        pickle.dump(standardize_feature_transformer, open(path, 'wb'))
    else:
        raise ValueError('Preprocessing method not supported')

    # Concatenate numerical and categorical features
    if NO_CAT:
        actions = numerical_actions.to_numpy()
    else:
        actions = pd.concat([numerical_actions, categorical_actions], axis = 1).to_numpy()
    observations = observations.to_numpy()

    return actions, observations


def visualize_data(data, exp_path):
    visualize_path = os.path.join(exp_path, 'visualize')
    if not os.path.exists(visualize_path):
        os.makedirs(visualize_path)

    fig, ax = plt.subplots(data.shape[1], 2)
    
    lambda_values = []
    for i in range(data.shape[1]):
        sns.distplot(data, hist=True, kde=True, kde_kws={'shade': True, 'linewidth': 2},
                 label='Non-Normal', color='green', ax=ax[i,0])
    
        fitted_data, fitted_lambda = stats.boxcox(data.iloc[:,i])
        lambda_values.append(fitted_lambda)

        sns.distplot(fitted_data, hist=True, kde=True, kde_kws={'shade': True, 'linewidth': 2},
                 label='Non-Normal', color='green', ax=ax[i,1])
    

    f = open(os.path.join(visualize_path, 'data_visualization.txt'), 'w')

    for i in range(len(lambda_values)):
        print('Lambda value used for Transformation in {} Sample {}'.format(list(data.columns)[i], lambda_values[i]))
        f.write('Lambda value used for Transformation in {} Sample {}\n'.format(list(data.columns)[i], lambda_values[i]))

    f.close()

    plt.legend(loc='upper right')
    fig.set_figheight(6)
    fig.set_figwidth(15)
    # Save the figure
    fig.savefig(os.path.join(visualize_path, 'data_visualization.png'))
    # Show the figure autoclose after 5 seconds
    plt.show(block=False)
    plt.pause(5)
    plt.close()


def main(_):
    # Define the experiment folder to save the model
    exp_name = 'mlp'
    exp_path = os.path.join(FLAGS.model_path, exp_name)
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)

    # Load the data
    if FLAGS.custom_dataset:
        actions_path = os.path.join(FLAGS.data_path, 'actions_feasible.csv')
        observations_path = os.path.join(FLAGS.data_path, 'observations_feasible.csv')
        actions = pd.read_csv(actions_path)
        observations = pd.read_csv(observations_path)
    else:
        california = fetch_california_housing()
        actions = pd.DataFrame(california.data, columns=california.feature_names)
        observations = pd.DataFrame(california.target, columns=['MEDV'])
    
    output = observations.copy()
    if FLAGS.output_index >= output.shape[1]:
        raise ValueError('Output index is out of range')
    output = output.iloc[:, FLAGS.output_index]

    observations = observations.loc[:, (observations != observations.iloc[0]).any()]

    X, y = preprocess_data(actions, output, exp_path)

    # Visualize the data
    if FLAGS.visualize:
        visualize_data(observations, exp_path)

    # Train the model
    if FLAGS.train:
        print('------Training the model------')
        
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
        
        # Split the data into train and test
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=FLAGS.train_size, random_state=FLAGS.seed)
        X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, train_size=0.5, random_state=FLAGS.seed)

        if FLAGS.activation == 'relu':
            activation = nn.ReLU()
        elif FLAGS.activation == 'tanh':
            activation = nn.Tanh()
        elif FLAGS.activation == 'sigmoid':
            activation = nn.Sigmoid()
        elif FLAGS.activation == 'LeakyRelu':
            activation = nn.LeakyReLU()
        elif FLAGS.activation == 'ELU':
            activation = nn.ELU()
        else:
            raise ValueError('Activation function not supported')
        
        if FLAGS.output_activation == 'softmax':
            output_activation = nn.Softmax()
        elif FLAGS.output_activation == 'sigmoid':
            output_activation = nn.Sigmoid()
        else:
            raise ValueError('Output activation function not supported')
        
        layers = []
        common_ratio = X.shape[1]**float(1/(FLAGS.n_hidden_layers + 1))
        neurons = X.shape[1]
        
        for i in range(FLAGS.n_hidden_layers):
            layers.append(nn.Linear(int(neurons), int(neurons/common_ratio)))
            layers.append(activation)
            neurons = float(neurons/common_ratio)
            
        layers.append(nn.Linear(int(neurons), 1))
            
        # Define the model
        regressor = nn.Sequential(*layers)

        if FLAGS.loss == 'mse':
            loss_fn = nn.MSELoss()
        elif FLAGS.loss == 'l1':
            loss_fn = nn.L1Loss()
        elif FLAGS.loss == 'humber':
            loss_fn = nn.HumberLoss()
        else:
            raise ValueError('Loss function not supported')
        
        if FLAGS.optimizer == 'adam':
            optimizer = optim.Adam(regressor.parameters(), lr=FLAGS.learning_rate)
        elif FLAGS.optimizer == 'sgd':
            optimizer = optim.SGD(regressor.parameters(), lr=FLAGS.learning_rate)
        elif FLAGS.optimizer == 'rmsprop':
            optimizer = optim.RMSprop(regressor.parameters(), lr=FLAGS.learning_rate)
        elif FLAGS.optimizer == 'LBFGS':
            optimizer = optim.LBFGS(regressor.parameters(), lr=FLAGS.learning_rate)
        else:
            raise ValueError('Optimizer not supported')
        
        best_loss = np.inf
        patience = FLAGS.patience
        best_params = None
        history_train, history_val = [], []
        train_loss = 0
        # Train the model
        for epoch in range(FLAGS.n_epochs):          
            # Train the model
            train_loss = 0
            for i in range(0, X_train.shape[0], FLAGS.batch_size):
                # Forward pass
                y_pred = regressor(X_train[i:i+FLAGS.batch_size])
                # Compute loss
                loss = loss_fn(y_pred, y_train[i:i+FLAGS.batch_size])
                train_loss += loss.item()
                # Zero gradients
                optimizer.zero_grad()
                # Backward pass
                loss.backward()
                # Update weights
                optimizer.step()
            # Evaluate the model validation dataset
            history_train.append(train_loss/X_train.shape[0])
            with torch.no_grad():
                y_pred = regressor(X_val)
                loss = loss_fn(y_pred, y_val)
                history_val.append(loss.item())

            # Early stopping
            if FLAGS.early_stopping:
                if patience == 0:
                    print('Early stopping at epoch {}'.format(epoch))
                    break
                else:
                    if best_loss < loss.item():
                        patience -= 1
                    else:
                        best_loss = loss.item()
                        best_params = regressor.state_dict()
                        patience = FLAGS.patience

        regressor.load_state_dict(best_params)
        # Evaluate the model for train dataset
        with torch.no_grad():
            y_pred = regressor(X_train)
        mse_train = mse(y_train, y_pred)
        print('MSE on train set: {}'.format(mse_train))

        # Evaluate the model for test dataset
        with torch.no_grad():
            y_pred = regressor(X_test)
        mse_test = mse(y_test, y_pred)
        print('MSE on test set: {}'.format(mse_test))

        # Visualize the loss
        plt.figure(figsize=(8, 6))
        plt.plot(history_train, label='train')
        plt.plot(history_val, label='val')
        plt.legend()
        plt.savefig(os.path.join(exp_path, 'loss_graph_{}.png'.format(FLAGS.output_index)))
        plt.show(block=False)
        plt.pause(5)
        plt.close()

        # Visualize the results
        y_test_series = pd.Series(y_test.reshape(-1))
        y_pred_series = pd.Series(y_pred.reshape(-1))
        results_df = pd.DataFrame()
        results_df['observation-{}'.format(FLAGS.output_index)] = y_test_series
        results_df['observation-{}-predicted'.format(FLAGS.output_index)] = y_pred_series

        plt.figure(figsize=(8, 6))
        sns.scatterplot(x='observation-{}'.format(FLAGS.output_index), y='observation-{}-predicted'.format(FLAGS.output_index),
                        data=results_df)
        sns.regplot(x='observation-{}'.format(FLAGS.output_index), y='observation-{}-predicted'.format(FLAGS.output_index),
                        data=results_df, color='orange', scatter=False)
        plt.savefig(os.path.join(exp_path, 'results_graph_{}.png'.format(FLAGS.output_index)))
        plt.show(block=False)
        plt.pause(5)
        plt.close()
        # Save the model
        path = os.path.join(exp_path, 'model_{}.joblib'.format(FLAGS.output_index))
        pickle.dump(regressor, open(path, 'wb'))

        FLAGS.append_flags_into_file(os.path.join(exp_path, 'flags_{}.txt'.format(FLAGS.output_index)))

        loaded_regressor = pickle.load(open(path, 'rb'))
        y_pred = loaded_regressor(X_test)
        mse_test_load = mse(y_test, y_pred.detach().numpy())

        # Check if the model is saved correctly
        if mse_test == mse_test_load:
            print('Models saved successfully at {}'.format(path))
        else:
            raise Exception('Model is not saved correctly')


if __name__ == '__main__':
    app.run(main)