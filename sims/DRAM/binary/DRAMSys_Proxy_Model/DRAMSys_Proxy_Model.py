import os
import pickle
import joblib
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import mean_squared_error
from scipy.special import inv_boxcox

class DRAMSysProxyModel:

  def __init__(self) -> None:
    self.model_directory = os.path.join(os.path.dirname(__file__), 'models')
    self.transformer_directory = os.path.join(os.path.dirname(__file__), 'transformers')
    self.energy_model_name = 'energy_model.pkl'
    self.power_model_name = 'power_model.pkl'
    self.latency_model_name = 'latency_model.pkl'
    self.numeric_data_transformer_name = 'numeric_data_transformer.pkl'
    self.target_data_transformer_name = 'target_data_transformer.pkl'
    self.one_hot_encoder_name = 'one_hot_encoder.pkl'
    self.label_encoder_name_list = ['label_encoder_Arbiter.pkl',
                                    'label_encoder_PagePolicy.pkl',
                                    'label_encoder_RefreshPolicy.pkl',
                                    'label_encoder_RespQueue.pkl',
                                    'label_encoder_Scheduler.pkl',
                                    'label_encoder_SchedulerBuffer.pkl']
    self.latency_lambda = 0.0

  def run_proxy_model(self, actions):

    '''Load models, transforms features, predicts energy/power/latency, 
        inverse transforms predictions, then returns prediction.
    
      Args:
        energy_model_name (str): file name of energy model 
        latency_model_name (str): file name of latency model
        power_model_name (str): file name of power model
        actions (pd.DataFrame): dataframe of features used to make a prediction
        numeric_data_transformer_name (str): file name holding numeric transformer data
        target_data_transformer_name (str): file name holding target transformer data
        one_hot_encoder_name (str): file name holding one hot encoder transformer data
        label_encoder_name_list (list): list holding file names holding label encoder
          transformer for each categorical feature. Must be passed in this order:
            0.) 'Arbiter' 1.) 'PagePolicy' 2.) 'RefreshPolicy' 3.) 'RespQueue' 4.) 'Scheduler' 5.) 'SchedulerBuffer'
        model_directory (str): directory name holding model files
        transformer_directory (str): directory name holding model data transformers
        latency_lambda (float): boxcox lambda value used to transform latency

      Returns:
        predicted: pd.DataFrame consisting of Energy, Power, and Latency columns
    '''
    # Set Categorical and Numerical Variables
    categorical_variables = ['Arbiter','PagePolicy','RefreshPolicy', 'RespQueue', 'Scheduler','SchedulerBuffer']
    numerical_variables = ['MaxActiveTransactions', 'RefreshMaxPostponed', 'RefreshMaxPulledin', 'RequestBufferSize']	

    # Create path to model
    energy_path = os.path.join(self.model_directory, self.energy_model_name)
    power_path = os.path.join(self.model_directory, self.power_model_name)
    latency_path = os.path.join(self.model_directory, self.latency_model_name)

    # Load Models
    energy_model = pickle.load(open(energy_path, 'rb'))
    power_model = pickle.load(open(power_path, 'rb'))
    latency_model = pickle.load(open(latency_path, 'rb'))

    #Categorical Data
    assert (self.one_hot_encoder_name is not None) or (self.label_encoder_name_list is not None), 'Must pass in Categorical Data Encoder File'
    if self.one_hot_encoder_name is not None:
      # Load One Hot Encoder
      encoding_path = os.path.join(self.transformer_directory, self.one_hot_encoder_name)
      enc = pickle.load(open(encoding_path, 'rb'))
      # One Hot Encode Categorical Variables
      X_categorical_data = enc.transform(actions[categorical_variables]).toarray()
      X_categorical_data = pd.DataFrame(X_categorical_data)
    elif self.label_encoder_name_list is not None: 
      X_categorical_data = pd.DataFrame()
      for label_encoder_name, categorical_variable in zip(self.label_encoder_name_list, categorical_variables):
        # Load Label Encoder
        encoding_path = os.path.join(self.transformer_directory, label_encoder_name) 
        enc = pickle.load(open(encoding_path, 'rb'))
        # Label Encode Categorical Variable
        encoded_categorical_data = enc.transform(actions[categorical_variable])
        # Add label encoded column to categorical df
        X_categorical_data[categorical_variable] = encoded_categorical_data

    # Transform Numerical Data
    if not self.numeric_data_transformer_name == None:
      # Get Transformer Path
      numerical_data_transformer = os.path.join(self.transformer_directory, self.numeric_data_transformer_name)
      # Load Numerical Transformer
      standard_scaler_numerical_transformer = joblib.load(numerical_data_transformer)
      # Transform Numberical Features
      X_numerical_data = pd.DataFrame(standard_scaler_numerical_transformer.transform(actions[numerical_variables]), columns=[numerical_variables])
    else:
      X_numerical_data = actions[numerical_variables]
    
    # Join Numerical and Categorical Features
    X = X_numerical_data.join(X_categorical_data)

    # Predict Energy
    energy = energy_model.predict(X)
    # Predict Power
    power = power_model.predict(X)
    # Predict Latency
    latency = latency_model.predict(X)

    # Predictions
    predicted = pd.DataFrame({'Energy': energy, 'Power': power, 'Latency': latency})

    # Inverse Transform Prediction 
    if not self.target_data_transformer_name == None:
      # Get Transformer Path
      target_data_transformer = os.path.join(self.transformer_directory, self.target_data_transformer_name)
      # Load Numerical Transformer
      standard_scaler_target_transformer = joblib.load(target_data_transformer)
      # Transform Numberical Features
      predicted = standard_scaler_target_transformer.inverse_transform(predicted)
      predicted = pd.DataFrame(predicted, columns=['Energy', 'Power', 'Latency'])

    # Inverse Boxcox Transform Latency
    if not self.latency_lambda == None:
      predicted['Latency'] = inv_boxcox(predicted['Latency'], self.latency_lambda).tolist()

    return predicted

# For testing
if __name__ == '__main__':
    example_df_path = os.path.join('data', 'Example_DRAMSys_Proxy_Model_Data.csv')
    df = pd.read_csv(example_df_path)

    df_y = df[['Energy','Power','Latency']]
    proxy_model = DRAMSysProxyModel()
    example_y_pred = proxy_model.run_proxy_model(df)

    print('Energy Error:', mean_squared_error(df_y['Energy'], example_y_pred['Energy']))
    print('Power Error:', mean_squared_error(df_y['Power'], example_y_pred['Power']))
    print('Latency Error:', mean_squared_error(df_y['Latency'], example_y_pred['Latency']))