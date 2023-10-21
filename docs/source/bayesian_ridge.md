# Bayesian Ridge Regression

## Example Usage
An example command that can be passed is as follows:
```shell:
python train_bayesian_ridge.py --train=True --n_iter=250
```

This page provides documentation for the Bayesian Ridge Regression code. The code includes data preprocessing, model training, and visualization. Each section below corresponds to a specific part of the code.

## Define Parameters
In this section, we define various parameters used throughout the code. These parameters include file paths, preprocessing options, encoding methods, and hyperparameters for the Bayesian Ridge model. These parameters can be passed as flags.

```python:
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
flags.DEFINE_bool('custom_dataset', False, 'Wheter to use a custom dataset or not')

# Hyperparameters for the model
flags.DEFINE_integer('n_iter', 300, 'Maximum number of iterations. The algorithm will converge if it reaches this number of iterations.')
flags.DEFINE_float('tol', 1e-3, 'Precision of the solution.')
flags.DEFINE_float('alpha_1', 1e-6, 'Hyper-parameter : shape parameter for the Gamma distribution prior over the alpha parameter.')
flags.DEFINE_float('alpha_2', 1e-6, 'Hyper-parameter : inverse scale parameter (rate parameter) for the Gamma distribution prior over the alpha parameter.')
flags.DEFINE_float('lambda_1', 1e-6, 'Hyper-parameter : shape parameter for the Gamma distribution prior over the lambda parameter.')
flags.DEFINE_float('lambda_2', 1e-6, 'Hyper-parameter : inverse scale parameter (rate parameter) for the Gamma distribution prior over the lambda parameter.')
flags.DEFINE_float('alpha_init', None, 'Initial value for alpha (precision of the noise). If not set, alpha_init is 1/Var(y).')
flags.DEFINE_float('lambda_init', None, 'Initial value for lambda (precision of the weights). If not set, lambda_init is 1.')
flags.DEFINE_bool('compute_score', False, 'If True, compute the objective function at each step of the model. Useful for plotting the evolution of the solution.')
flags.DEFINE_bool('fit_intercept', True, 'Whether to calculate the intercept for this model. If set to false, no intercept will be used in calculations (e.g. data is expected to be already centered).')
flags.DEFINE_bool('copy_X', True, 'If True, X will be copied; else, it may be overwritten.')
flags.DEFINE_bool('verbose', False, 'Verbose mode when fitting the model.')

FLAGS = flags.FLAGS
```

## Data Preprocessing
### Preprocess Data Function
The preprocess_data function handles data preprocessing tasks, including encoding categorical features and normalizing numerical features.
```python:
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

```

### Visualize Data Function
The visualize_data function visualizes the data distribution and performs a Box-Cox transformation.
```python:
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
```
## Main Function
### Loading Data
In the main function, data is loaded either from a custom dataset or the California housing dataset.

```python:
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
```
### Data preprocessing
Data preprocessing is performed using the preprocess_data function.
```python:
X, y = preprocess_data(actions, output, exp_path)
```
### Data Visualization
Optional data visualization can be enabled by setting the visualize flag to 'True'.
```python:
# Visualize the data
if FLAGS.visualize:
    visualize_data(observations, exp_path)
```
### Model Training
If the train flag is set to True, the Bayesian Ridge regression model is trained.
```python:
if FLAGS.train:
        print('------Training the model------')
        # Split the data into train and test
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=FLAGS.train_size, random_state=FLAGS.seed)

        # Define the model
        regressor = BayesianRidge(n_iter=FLAGS.n_iter, tol=FLAGS.tol, alpha_1=FLAGS.alpha_1, alpha_2=FLAGS.alpha_2,
                                  lambda_1=FLAGS.lambda_1, lambda_2=FLAGS.lambda_2, alpha_init=FLAGS.alpha_init, 
                                  lambda_init=FLAGS.lambda_init, compute_score=FLAGS.compute_score, 
                                  fit_intercept=FLAGS.fit_intercept, copy_X=FLAGS.copy_X, verbose=FLAGS.verbose)
        
        # Train the model
        regressor.fit(X_train, y_train[:, 0])

        # Evaluate the model for train dataset
        y_pred = regressor.predict(X_train)
        mse_train = mse(y_train, y_pred)
        print('MSE on train set: {}'.format(mse_train))

        # Evaluate the model for test dataset
        y_pred = regressor.predict(X_test)
        mse_test = mse(y_test, y_pred)
        print('MSE on test set: {}'.format(mse_test))
```