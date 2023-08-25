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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# Define parameters for the training/handling of the data and model
flags.DEFINE_string('data_path', './data', 'Path to the data')
flags.DEFINE_string('model_path', './models', 'Path to the model')
flags.DEFINE_integer('seed', 123, 'Seed for the random number generator')
flags.DEFINE_float('train_size', 0.8, 'the split between train and test dataset')
flags.DEFINE_enum('preprocess', 'normalize', ['normalize', 'standardize'], 'Preprocessing method')
flags.DEFINE_enum('encode', 'one_hot', ['one_hot', 'label'], 'Encoding method')
flags.DEFINE_bool('visualize', False, 'enable visualization of the data')
flags.DEFINE_bool('train', False, 'enable training of the model')

# Hyperparameters for the model
flags.DEFINE_enum('penalty', 'l2', ['l1', 'l2', 'elasticnet', None], 'Used to specify the norm used in the penalization')
flags.DEFINE_bool('dual', False, 'Dual or primal formulation')
flags.DEFINE_float('tol', 0.0001, 'Tolerance for stopping criteria')
flags.DEFINE_float('C', 1.0, 'Inverse of regularization strength; must be a positive float')
flags.DEFINE_bool('fit_intercept', True, 'Specifies if a constant (a.k.a. bias or intercept) should be added to the decision function')
flags.DEFINE_float('intercept_scaling', 1.0, 'Useful only when the solver “liblinear” is used and self.fit_intercept is set to True. In this case, x becomes [x, self.intercept_scaling], i.e. a “synthetic” feature with constant value equal to intercept_scaling is appended to the instance vector')
flags.DEFINE_bool('class_weight', None, 'Weights associated with classes in the form {class_label: weight}. If not given, all classes are supposed to have weight one')
flags.DEFINE_integer('random_state', None, 'The seed of the pseudo random number generator to use when shuffling the data')
flags.DEFINE_enum('solver', 'lbfgs', ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'], 'Algorithm to use in the optimization problem')
flags.DEFINE_integer('max_iter', 100, 'Maximum number of iterations taken for the solvers to converge')
flags.DEFINE_enum('multi_class', 'auto', ['auto', 'ovr', 'multinomial'], 'If the option chosen is “ovr”, then a binary problem is fit for each label. For “multinomial” the loss minimised is the multinomial loss fit across the entire probability distribution, even when the data is binary. “multinomial” is unavailable when solver=’liblinear’. “auto” selects “ovr” if the data is binary, or if solver=’liblinear’, and otherwise selects “multinomial”')
flags.DEFINE_integer('verbose', 0, 'For the liblinear and lbfgs solvers set verbose to any positive number for verbosity')
flags.DEFINE_bool('warm_start', False, 'When set to True, reuse the solution of the previous call to fit as initialization, otherwise, just erase the previous solution')
flags.DEFINE_integer('n_jobs', None, 'Number of CPU cores used when parallelizing over classes if multi_class=’ovr’”. This parameter is ignored when the solver is set to “liblinear” regardless of whether “multi_class” is specified or not. None means 1 unless in a joblib.parallel_backend context. -1 means using all processors')
flags.DEFINE_float('l1_ratio', None, 'The Elastic-Net mixing parameter, with 0 <= l1_ratio <= 1. Only used if penalty is “elasticnet”. Setting l1_ratio=0 is equivalent to using penalty=”l2”, while setting l1_ratio=1 is equivalent to using penalty=”l1”. For 0 < l1_ratio <1, the penalty is a combination of L1 and L2')

FLAGS = flags.FLAGS

def preprocess_data(actions, observations, exp_path):
    # Categorical features
    categorical_cols = list(set(actions.columns) - set(actions._get_numeric_data().columns))
    categorical_actions = actions[categorical_cols]
    
    # Numerical features
    numerical_actions = actions._get_numeric_data()
    
    encoder_path = os.path.join(exp_path, 'encoder')
    if not os.path.exists(encoder_path):
        os.makedirs(encoder_path)

    # Encode categorical features
    if FLAGS.encode == 'one_hot':
        # One-hot encode categorical features
        enc = OneHotEncoder(handle_unknown='ignore')
        enc.fit(categorical_actions)
        # Save the encoder
        path = os.path.join(encoder_path, 'one_hot_encoder.joblib')
        pickle.dump(enc, open(path, 'wb'))
        # Transform the categorical features
        dummy_col_names = pd.get_dummies(categorical_actions).columns
        categorical_actions = pd.DataFrame(enc.transform(categorical_actions).toarray(), columns=dummy_col_names)
    elif FLAGS.encode == 'label':
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
    else:
        raise ValueError('Encoding method not supported')

    preprocess_data_path = os.path.join(exp_path, 'preprocess_data')
    if not os.path.exists(preprocess_data_path):
        os.makedirs(preprocess_data_path)

    # Normalize numerical features
    if FLAGS.preprocess == 'normalize':
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
        path = os.path.join(preprocess_data_path, 'normalize_feature_transformer_observations.joblib')
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
        path = os.path.join(preprocess_data_path, 'standardize_feature_transformer_observations.joblib')
        pickle.dump(standardize_feature_transformer, open(path, 'wb'))
    else:
        raise ValueError('Preprocessing method not supported')

    # Concatenate numerical and categorical features
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
    # Constraints for the hyperparameters
    if FLAGS.precompute != 'auto':
        FLAGS.precompute = bool(FLAGS.precompute)
    
    # Define the experiment folder to save the model
    exp_name = 'logistic_regression'
    exp_path = os.path.join(FLAGS.model_path, exp_name)
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)

    # Load the data
    actions_path = os.path.join(FLAGS.data_path, 'actions_feasible.csv')
    observations_path = os.path.join(FLAGS.data_path, 'observations_feasible.csv')

    actions = pd.read_csv(actions_path)
    observations = pd.read_csv(observations_path)
    observations = observations.drop(['observation-2', 'observation-3', 'observation-4'], axis = 1)

    X, y = preprocess_data(actions, observations, exp_path)

    # Visualize the data
    if FLAGS.visualize:
        visualize_data(observations, exp_path)

    # Train the model
    if FLAGS.train:
        print('------Training the model------')
        # Split the data into train and test
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=FLAGS.train_size, random_state=FLAGS.seed)

        # Define the model
        reg = LogisticRegression(penalty=FLAGS.penalty, dual=FLAGS.dual, tol=FLAGS.tol, C=FLAGS.C, fit_intercept=FLAGS.fit_intercept, intercept_scaling=FLAGS.intercept_scaling, class_weight=FLAGS.class_weight, random_state=FLAGS.random_state, solver=FLAGS.solver, max_iter=FLAGS.max_iter, multi_class=FLAGS.multi_class, verbose=FLAGS.verbose, warm_start=FLAGS.warm_start, n_jobs=FLAGS.n_jobs, l1_ratio=FLAGS.l1_ratio)
        
        # Train the model
        reg.fit(X_train, y_train)

        # Evaluate the model for train dataset
        y_pred = reg.predict(X_train)
        mse_train = mse(y_train, y_pred)
        print('MSE on train set: {}'.format(mse_train))

        # Evaluate the model for test dataset
        y_pred = reg.predict(X_test)
        mse_test = mse(y_test, y_pred)
        print('MSE on test set: {}'.format(mse_test))

        # Save the model
        path = os.path.join(exp_path, 'model.joblib')
        pickle.dump(reg, open(path, 'wb'))

        FLAGS.append_flags_into_file(os.path.join(exp_path, 'flags.txt'))

        loaded_rf = pickle.load(open(path, 'rb'))
        y_pred = loaded_rf.predict(X_test)
        mse_test_load = mse(y_test, y_pred)

        # Check if the model is saved correctly
        if mse_test == mse_test_load:
            print('Model saved successfully at {}'.format(path))
        else:
            raise Exception('Model is not saved correctly')


if __name__ == '__main__':
    app.run(main)
