import os
import pickle

from absl import flags
from absl import app

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.datasets import fetch_california_housing

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
flags.DEFINE_bool('custom_dataset', False, 'enable training of the model on custom data')

# Hyperparameters for the model
flags.DEFINE_integer('n_estimators', 100, 'Number of trees in the forest')
flags.DEFINE_string('criterion', 'squared_error', 'Function to measure the quality of a split')
flags.DEFINE_integer('max_depth', None, 'Maximum depth of the tree')
flags.DEFINE_float('min_samples_split', 2, 'Minimum number of samples required to split an internal node')
flags.DEFINE_float('min_samples_leaf', 1, 'Minimum number of samples required to be at a leaf node')
flags.DEFINE_float('min_weight_fraction_leaf', 0.0, 'Minimum weighted fraction of the sum total of weights required to be at a leaf node')
flags.DEFINE_enum('max_features', None, ['sqrt', 'log2'], 'Number of features to consider when looking for the best split')
flags.DEFINE_integer('max_leaf_nodes', None, 'Grow trees with max_leaf_nodes in best-first fashion')
flags.DEFINE_float('min_impurity_decrease', 0.0, 'A node will be split if this split induces a decrease of the impurity greater than or equal to this value')
flags.DEFINE_bool('bootstrap', True, 'Whether bootstrap samples are used when building trees')
flags.DEFINE_bool('oob_score', False, 'Whether to use out-of-bag samples to estimate the R^2 on unseen data')
flags.DEFINE_integer('n_jobs', None, 'The number of jobs to run in parallel for both fit and predict')
flags.DEFINE_integer('random_state', None, 'Controls both the randomness of the bootstrapping of the samples used when building trees (if bootstrap=True) and the sampling of the features to consider when looking for the best split at each node (if max_features < n_features)')
flags.DEFINE_integer('verbose', 0, 'Controls the verbosity when fitting and predicting')
flags.DEFINE_bool('warm_start', False, 'When set to True, reuse the solution of the previous call to fit and add more estimators to the ensemble, otherwise, just fit a whole new forest')
flags.DEFINE_float('ccp_alpha', 0.0, 'Complexity parameter used for Minimal Cost-Complexity Pruning. The subtree with the largest cost complexity that is smaller than ccp_alpha will be chosen')
flags.DEFINE_float('max_samples', None, 'If bootstrap is True, the number of samples to draw from X to train each base estimator')

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
    # Constraints set on the hyperparameters
    if FLAGS.min_samples_leaf >= 1:
        FLAGS.min_samples_leaf = int(FLAGS.min_samples_leaf)
    if FLAGS.min_samples_split >= 1:
        FLAGS.min_samples_split = int(FLAGS.min_samples_split)

    # Define the experiment folder to save the model
    exp_name = 'random_forest'
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
        regressor = RandomForestRegressor(n_estimators=FLAGS.n_estimators, criterion=FLAGS.criterion, max_depth=FLAGS.max_depth,
                                min_samples_split=FLAGS.min_samples_split, min_samples_leaf=FLAGS.min_samples_leaf,
                                min_weight_fraction_leaf=FLAGS.min_weight_fraction_leaf, max_features=FLAGS.max_features,
                                          random_state=FLAGS.random_state, max_leaf_nodes=FLAGS.max_leaf_nodes,
                                          min_impurity_decrease=FLAGS.min_impurity_decrease, ccp_alpha=FLAGS.ccp_alpha)
        
        # Train the model
        regressor.fit(X_train, y_train)

        # Evaluate the model for train dataset
        y_pred = regressor.predict(X_train)
        mse_train = mse(y_train, y_pred)
        print('MSE on train set: {}'.format(mse_train))

        # Evaluate the model for test dataset
        y_pred = regressor.predict(X_test)
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

        # Save the model
        path = os.path.join(exp_path, 'model_{}.joblib'.format(FLAGS.output_index))
        pickle.dump(regressor, open(path, 'wb'))

        FLAGS.append_flags_into_file(os.path.join(exp_path, 'flags_{}.txt'.format(FLAGS.output_index)))

        loaded_regressor = pickle.load(open(path, 'rb'))
        y_pred = loaded_regressor.predict(X_test)
        mse_test_load = mse(y_test, y_pred)

        # Check if the model is saved correctly
        if mse_test == mse_test_load:
            print('Model saved successfully at {}'.format(path))
        else:
            raise Exception('Model is not saved correctly')


if __name__ == '__main__':
    app.run(main)
