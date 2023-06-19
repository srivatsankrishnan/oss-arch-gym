from sklearn.base import BaseEstimator

class CustomEstimator(BaseEstimator):
   
   def __init__(self, a_=1.0, b_=1, c_=1, d_='constant'):
      ''' 
      All the default values of the "model" should be initialized here. 
      '''
      self.a_ = a_
      self.b_ = b_
      self.c_ = c_
      self.d_ = d_
     
      
   def fit (self, X, y=None):
      '''
      Simple dummy function to fit 
      '''
      if self.d_ == "linear":
         y1 = self.a_*X[0]+ self.b_ + self.c_* self.c_
      elif self.d_ == "poly":
         y1 = self.a_*(X[0]*X[0])+ self.b_ + self.c_* self.c_
      else: # constant
         y1 = self.a_ + self.b_ + self.c_* self.c_
      
      return y1

   
   def predict(self, X, y=None):
      return NotImplementedError

   
   def score(self, X, y=None):
      return NotImplementedError


   def get_params(self, deep=False):
      return {'a_': self.a_,
              'b_': self.b_,
              'c_': self.c_,
              'd_': self.d_,
             }

   def set_params(self, **params):
      self.a_ = params['a_']
      self.b_ = params['b_']
      self.c_ = params['c_']
      self.d_ = params['d_']
      return self 


