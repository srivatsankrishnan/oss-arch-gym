import os
import pickle

from absl import flags
from absl import app
import lib_node as lib
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
from qhoptim.pyt import QHAdam

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
# Dense Block Parameters
flags.DEFINE_integer('layer_dim', 128, 'Dimension of the hidden layer')
flags.DEFINE_integer('num_layers', 6, 'Number of hidden layers')
flags.DEFINE_integer('tree_dim', 3, 'Dimension of the tree')
flags.DEFINE_integer('depth', 8, 'Depth of the tree')
flags.DEFINE_bool('flatten_output', False, 'Flatten the output of the tree')
flags.DEFINE_enum('choice_function', 'EntMax15', ['EntMax15', 'SparseMax', 'SoftMax'], 'Choice function to calculate feature weights')
flags.DEFINE_enum('bin_function', 'SparseMoid', ['SparseMoid', 'EntMoid15', 'SigMoid'], 'Function to calculate tree leaf weights')
flags.DEFINE_float('threshold_init_beta', 1.0, 'initializes threshold to a q-th quantile of data points')
flags.DEFINE_float('threshold_init_cutoff', 1.0, 'threshold log-temperatures initializer')

# Optimizer Parameters
flags.DEFINE_float('nu_lower', 0.7, 'Lower bound of the nu parameter')
flags.DEFINE_float('nu_upper', 1.0, 'Upper bound of the nu parameter')
flags.DEFINE_float('beta_lower', 0.95, 'Lower bound of the beta parameter')
flags.DEFINE_float('beta_upper', 0.998, 'Upper bound of the beta parameter')

flags.DEFINE_bool('verbose', True, 'Verbose mode')
flags.DEFINE_enum('device', 'cpu', ['cpu', 'cuda'], 'Device to train the model on')
flags.DEFINE_bool('warm_start', False, 'Warm start the model')
flags.DEFINE_integer('batch_size', 128, 'Batch size for training the model')
flags.DEFINE_integer('n_epochs', 1000, 'Number of epochs to train the model')

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
    if FLAGS.preprocess == None:
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
    if os.path.exists('./logs'):
        os.system('rm -rf ./logs')

    # Constraint on the Hyperparamters
    if FLAGS.choice_function == 'EntMax15':
        choice_function = lib.entmax15
    elif FLAGS.choice_function == 'SparseMax':
        choice_function = lib.sparsemax
    elif FLAGS.choice_function == 'SoftMax':
        choice_function = nn.Softmax
    else:
        raise ValueError('Choice function not supported')
    
    if FLAGS.bin_function == 'SparseMoid':
        bin_function = lib.sparsemoid
    elif FLAGS.bin_function == 'EntMoid15':
        bin_function = lib.entmoid15
    elif FLAGS.bin_function == 'SigMoid':
        bin_function = nn.Sigmoid
    else:
        raise ValueError('Bin function not supported')

    # Define the experiment folder to save the model
    exp_name = 'node'
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
        # Split the data into train and test
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=FLAGS.train_size, random_state=FLAGS.seed)

        # Define the model
        model = nn.Sequential(
            lib.DenseBlock(X.shape[1], 128, num_layers=6, tree_dim=3, depth=8, flatten_output=False,
                           choice_function=lib.entmax15, bin_function=lib.entmoid15),
            lib.Lambda(lambda x: x[..., 0].mean(dim=-1)),  # average first channels of every tree
        ).to(FLAGS.device)

        optimizer_params = {'nus':(FLAGS.nu_lower, FLAGS.nu_upper),
                            'betas':(FLAGS.beta_lower, FLAGS.beta_upper)}
        
        trainer = lib.Trainer(
            model=model, loss_function=nn.MSELoss(),
            experiment_name='checkpoints', warm_start=FLAGS.warm_start,
            Optimizer=QHAdam, optimizer_params=optimizer_params,
            verbose=FLAGS.verbose
        )
        
        best_mse = np.inf

        for batch in lib.iterate_minibatches(np.float32(X_train), np.float32(y_train), batch_size=FLAGS.batch_size,
                                             shuffle=True, epochs=FLAGS.n_epochs):
            metrics = trainer.train_on_batch(*batch, device=FLAGS.device)

            if metrics['loss'] < best_mse:
                best_mse = metrics['loss']
                trainer.save_checkpoint('model_{}'.format(FLAGS.output_index))

        # Evaluate the model for train dataset
        trainer.load_checkpoint(tag='model_{}'.format(FLAGS.output_index))
        y_pred = model(torch.from_numpy(X_train).float().to(FLAGS.device)).detach().cpu().numpy()
        mse_train = mse(y_train, y_pred)
        print('MSE on train set: {}'.format(mse_train))

        # Evaluate the model for test dataset
        y_pred = model(torch.from_numpy(X_test).float().to(FLAGS.device)).detach().cpu().numpy()
        mse_test = mse(y_test, y_pred)
        print('MSE on test set: {}'.format(mse_test))

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

        FLAGS.append_flags_into_file(os.path.join(exp_path, 'flags_{}.txt'.format(FLAGS.output_index)))


if __name__ == '__main__':
    app.run(main)
