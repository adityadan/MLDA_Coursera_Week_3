import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoLarsCV

data = pd.read_csv("Cat_stats.csv")

data_clean = data.dropna()
  
predvar = data_clean[['Body_length', 'Tail_length', 'Height', 'Tail_texture', 'Coat_colour']]

target = data_clean.Weight
 
predictors = predvar.copy()
from sklearn import preprocessing
predictors['Body_length'] = preprocessing.scale(predictors['Body_length'].astype('float64'))
predictors['Tail_length'] = preprocessing.scale(predictors['Tail_length'].astype('float64'))
predictors['Height'] = preprocessing.scale(predictors['Height'].astype('float64'))
predictors['Tail_texture'] = preprocessing.scale(predictors['Tail_texture'].astype('float64'))
predictors['Coat_colour'] = preprocessing.scale(predictors['Coat_colour'].astype('float64'))

pred_train, pred_test, tar_train, tar_test = train_test_split(predictors, target, test_size = 0.3, random_state = 50)

model = LassoLarsCV(cv = 5, precompute = False).fit(pred_train, tar_train)

dict(zip(predictors.columns, model.coef_))

m_log_alphas = -np.log10(model.alphas_)
ax = plt.gca()
plt.plot(m_log_alphas, model.coef_path_.T)
plt.axvline(-np.log10(model.alpha_), linestyle = '--', color = 'k', label = 'alpha CV')
plt.ylabel('Regression Coefficients')
plt.xlabel('-log(alpha)')
plt.title('Regression Coefficients Progression for Lasso Paths - Cat Data')

#m_log_alphascv = -np.log10(model.cv_alphas_)
#plt.figure()
#plt.plot(m_log_alphascv, model.cv_mse_path_, ':')
#plt.plot(m_log_alphascv, model.cv_mse_path_.mean(axis = -1), 'k', label = 'Average across the folds', linewidth = 2)
#plt.axvline(-np.log10(model.alpha_), linestyle = '--', color = 'k', label = 'alpha CV')
#plt.legend()
#plt.xlabel('-log(alpha)')
#plt.ylabel('Mean squared error')
#plt.title('Mean squared error on each fold')
         

from sklearn.metrics import mean_squared_error
train_error = mean_squared_error(tar_train, model.predict(pred_train))
test_error = mean_squared_error(tar_test, model.predict(pred_test))
print ('training data MSE')
print(train_error)
print ('test data MSE')
print(test_error)

rsquared_train = model.score(pred_train, tar_train)
rsquared_test = model.score(pred_test, tar_test)
print ('training data R-square')
print(rsquared_train)
print ('test data R-square')
print(rsquared_test)
