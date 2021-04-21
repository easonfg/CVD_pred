import pickle5 as pickle
from sksurv.ensemble import RandomSurvivalForest
from sksurv.datasets import load_whas500
from lifelines.utils import concordance_index
from sklearn.utils import resample

# from sklearn.externals import joblib
from sklearn import externals
#!pip install -U -q joblib
import joblib
import numpy as np
import pandas as pd

def c_index_bootstrap(T, E, scores, alpha=0.05, n_bootstrap=10, random_state=None):
  np.random.seed(random_state)

  c_index = concordance_index(T, -scores, E)

  stacked_arr = np.stack((T, -scores, E), axis=-1)
  c_idx_boot = np.array([concordance_index(*resample(stacked_arr).T) for _ in range(n_bootstrap)])

  c_index_boot = c_idx_boot.mean()
  sigma2 = c_idx_boot.var()
  CI_lower = np.quantile(c_idx_boot, alpha/2)
  CI_upper = np.quantile(c_idx_boot, 1 - alpha/2)

  return c_index, c_index_boot, sigma2, CI_lower, CI_upper


#X, y = load_whas500()
#y
#estimator = RandomSurvivalForest(n_estimators= 500, verbose = 1, n_jobs = -1)
#estimator.fit(X, y)
#joblib_file = "joblib_model.pkl"
#joblib.dump(estimator, joblib_file)

#E_test = np.loadtxt('../RS_composite_all.E_test.txt')
#T_test = np.loadtxt('../RS_composite_all.T_test.txt')
#E_train = np.loadtxt('../RS_composite_all.E_train.txt')
#T_train = np.loadtxt('../RS_composite_all.T_train.txt')
#with open('../RS_composite_all_X_TRAIN', 'rb') as pickle_file:
#    X_train = pickle.load(pickle_file)
#    with open('../RS_composite_all_X_test', 'rb') as pickle_file:
#        X_test = pickle.load(pickle_file)
#
#RS_y = np.array(list(tuple(zip(E_train, T_train))), dtype = [('E', bool), ('T', float)])
#RS_test_y = np.array(list(tuple(zip(E_test, T_test))), dtype = [('E', bool), ('T', float)])
#
##### make model
###rsf = RandomSurvivalForest(n_estimators = 1, verbose = 1, n_jobs=-1)
###rsf = RandomSurvivalForest(n_estimators = 30, verbose = 1, n_jobs=15)
#rsf = RandomSurvivalForest(n_estimators = 100, verbose = 1, n_jobs=-1)
##rsf = RandomSurvivalForest(n_estimators = 1000, verbose = 1, n_jobs=-2)
#rsf.fit(X_train, RS_y)
#joblib.dump(rsf, "joblib_model_composite_all_rsf100n.pkl")
#### make model
#
##jobModel = joblib.load('joblib_model_rsf1000n.pkl')
#
#testPred = pd.Series(rsf.predict(X_test))
#trainPred = pd.Series(rsf.predict(X_train))
#print('ALL test')
#print(c_index_bootstrap(T_test, E_test, testPred))
#print('ALL train')
#print(c_index_bootstrap(T_train, E_train, trainPred))
#
#
##sans base
#E_test = np.loadtxt('../RS_composite_sans_base.E_test.txt')
#T_test = np.loadtxt('../RS_composite_sans_base.T_test.txt')
#E_train = np.loadtxt('../RS_composite_sans_base.E_train.txt')
#T_train = np.loadtxt('../RS_composite_sans_base.T_train.txt')
#with open('../RS_composite_sans_base_X_TRAIN', 'rb') as pickle_file:
#    X_train = pickle.load(pickle_file)
#with open('../RS_composite_sans_base_X_test', 'rb') as pickle_file:
#    X_test = pickle.load(pickle_file)
#
#RS_y = np.array(list(tuple(zip(E_train, T_train))), dtype = [('E', bool), ('T', float)])
#RS_test_y = np.array(list(tuple(zip(E_test, T_test))), dtype = [('E', bool), ('T', float)])
#
##### make model
###rsf = RandomSurvivalForest(n_estimators = 1, verbose = 1, n_jobs=-1)
###rsf = RandomSurvivalForest(n_estimators = 30, verbose = 1, n_jobs=15)
#
#rsf = RandomSurvivalForest(n_estimators = 100, verbose = 1, n_jobs=-1)
##rsf = RandomSurvivalForest(n_estimators = 500, verbose = 1, n_jobs=-1)
#
##rsf = RandomSurvivalForest(n_estimators = 1000, verbose = 1, n_jobs=-2)
#rsf.fit(X_train, RS_y)
##joblib.dump(rsf, "joblib_model_composite_sans_base_rsf500n.pkl")
#joblib.dump(rsf, "joblib_model_composite_sans_base_rsf100n.pkl")
#### make model
#
##jobModel = joblib.load('joblib_model_rsf1000n.pkl')
#
#
#testPred = pd.Series(rsf.predict(X_test))
#trainPred = pd.Series(rsf.predict(X_train))
#print('sans base test')
#print(c_index_bootstrap(T_test, E_test, testPred))
#print('sans base train')
#print(c_index_bootstrap(T_train, E_train, trainPred))
#
#
#
#### lab
#E_test = np.loadtxt('../RS_composite_lab.E_test.txt')
#T_test = np.loadtxt('../RS_composite_lab.T_test.txt')
#E_train = np.loadtxt('../RS_composite_lab.E_train.txt')
#T_train = np.loadtxt('../RS_composite_lab.T_train.txt')
#with open('../RS_composite_lab_X_TRAIN', 'rb') as pickle_file:
#    X_train = pickle.load(pickle_file)
#    with open('../RS_composite_lab_X_test', 'rb') as pickle_file:
#        X_test = pickle.load(pickle_file)
#
#RS_y = np.array(list(tuple(zip(E_train, T_train))), dtype = [('E', bool), ('T', float)])
#RS_test_y = np.array(list(tuple(zip(E_test, T_test))), dtype = [('E', bool), ('T', float)])
#
##### make model
###rsf = RandomSurvivalForest(n_estimators = 1, verbose = 1, n_jobs=-1)
###rsf = RandomSurvivalForest(n_estimators = 30, verbose = 1, n_jobs=15)
#rsf = RandomSurvivalForest(n_estimators = 100, verbose = 1, n_jobs=-1)
##rsf = RandomSurvivalForest(n_estimators = 1000, verbose = 1, n_jobs=-2)
#rsf.fit(X_train, RS_y)
#joblib.dump(rsf, "joblib_model_composite_lab_rsf100n.pkl")
#### make model
#
##jobModel = joblib.load('joblib_model_rsf1000n.pkl')
#
#
#testPred = pd.Series(rsf.predict(X_test))
#trainPred = pd.Series(rsf.predict(X_train))
#print('lab test')
#print(c_index_bootstrap(T_test, E_test, testPred))
#print('lab train')
#print(c_index_bootstrap(T_train, E_train, trainPred))

#### lab
E_test = np.loadtxt('../RS_cvd_lab_demo.E_test.txt')
T_test = np.loadtxt('../RS_cvd_lab_demo.T_test.txt')
E_train = np.loadtxt('../RS_cvd_lab_demo.E_train.txt')
T_train = np.loadtxt('../RS_cvd_lab_demo.T_train.txt')
with open('../RS_cvd_lab_demo_X_TRAIN', 'rb') as pickle_file:
    X_train = pickle.load(pickle_file)
    with open('../RS_cvd_lab_demo_X_test', 'rb') as pickle_file:
        X_test = pickle.load(pickle_file)

RS_y = np.array(list(tuple(zip(E_train, T_train))), dtype = [('E', bool), ('T', float)])
RS_test_y = np.array(list(tuple(zip(E_test, T_test))), dtype = [('E', bool), ('T', float)])

#### make model
##rsf = RandomSurvivalForest(n_estimators = 1, verbose = 1, n_jobs=-1)
##rsf = RandomSurvivalForest(n_estimators = 30, verbose = 1, n_jobs=15)
rsf = RandomSurvivalForest(n_estimators = 100, verbose = 1, n_jobs=-1)
#rsf = RandomSurvivalForest(n_estimators = 1000, verbose = 1, n_jobs=-2)
rsf.fit(X_train, RS_y)
joblib.dump(rsf, "joblib_model_cvd_lab_demo_rsf100n.pkl")
### make model

#jobModel = joblib.load('joblib_model_rsf1000n.pkl')


testPred = pd.Series(rsf.predict(X_test))
trainPred = pd.Series(rsf.predict(X_train))
print('lab_demo test')
print(c_index_bootstrap(T_test, E_test, testPred))
print('lab_demo train')
print(c_index_bootstrap(T_train, E_train, trainPred))
