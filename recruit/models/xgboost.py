# for server
import matplotlib
matplotlib.use("agg")

import xgboost as xgb
from sklearn.metrics import make_scorer, explained_variance_score
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV

from recruit import datasets
from recruit.utils import unique_id_utils as uiu
from recruit.preprocess import feature_engineering as fe
from recruit.utils import submission_utils as su
from sklearn.base import BaseEstimator, RegressorMixin




class XGBoostReg(BaseEstimator, RegressorMixin):
    def __init__(self, n=30, step=1, unique_id='air_00a91d42b08b08d9', \
                    n_estimators=300, subsample=0.7, max_depth=15):

        # I need passing arguments directoly not **params due to the GridsearchCV.
        print("now initializing model.") # for debug
        self.n = n
        self.step = step
        self.unique_id = unique_id
        self.n_estimators = n_estimators
        self.subsample = subsample
        self.max_depth = max_depth

        self._init_model()

    # this __init__ is not conpatible for sklearn gridsearchCV.
    # def __init__(self, **params):
    #     self._init_model()
    #     self.set_params(**params)

    def set_params(self, **params):
        self.n = params.pop('n', 30)
        self.step = params.pop('step', 1)
        self.unique_id = params.pop('unique_id')
        self.n_estimators = params.pop('n_estimators', 300)
        self.subsample = params.pop('subsample', 0.7)
        self.max_depth = params.pop('max_depth', 15)

        if not len(params) == 0:
            raise ValueError(f'Unknown parameter {params.keys()}')

        return self

    def _init_model(self):
        # modify to change parameter and I can use CV
        self.model = xgb.XGBRegressor(n_estimators=self.n_estimators, learning_rate=0.08, gamma=0, subsample=self.subsample,
                           colsample_bytree=1, max_depth=self.max_depth, alpha=10, reg_lambda=1.)

    def fit(self, X, y):
        print("now fitting model.") # for debug
        self.model = self.model.fit(X, y)
        return self

    def _predict(self, X):
        predictions = self.model.predict(X)
        return predictions

    def eval(self, predictions, y_test):
        score = explained_variance_score(predictions, y_test)
        print(score)

    def _cut_under_zero(self, predictions):
        if isinstance(predictions, list):
            predictions = np.array(predictions)
        predictions[predictions <= 0] = 0.
        return predictions

    def predict(self, X_test):
        # not use X_test itself, but use just length.
        print("now predicting test data.") # for debug
        self.prediter = XGBoostPredictIter(self)
        predictions = self.prediter.predict_iter(X_test)
        return predictions


class XGBoostPredictIter(object):
    def __init__(self, xgboostreg):
        """
        xgboostreg : XGBoostReg
            after train finished
        """
        # set model parameters
        self.xgboostreg = xgboostreg
        self.unique_id = self.xgboostreg.unique_id
        self.n = self.xgboostreg.n
        self.step = self.xgboostreg.step

        # get train data
        self.data = datasets.RecruitDatasets()
        self.uih = uiu.UniqueIDHandler(unique_id=self.unique_id, data=self.data)
        self.df = self.uih.get_dataframe()

        # get air_reservation_data to make use of reservation data in test dataset.
        self.ar_subset = self.uih.arh.get_df_subset(self.unique_id)
        self.ar_df = self.uih.arh.get_visitdate_visiters_num_df(self.ar_subset)

    def predict_iter(self, X_test): 
        pred_len = X_test.shape[0] # not use X_test itself, but use just length.
        df_train = self.df
        predictions = []
        for _ in range(pred_len):
            f_eng_train = fe.FeatureEngineearing(self.n, self.step, df_train)
            X_next_all, _ = f_eng_train.get_X_y()

            X_last = X_next_all[-1]
            X_last = X_last.reshape(1, X_last.shape[0])

            prediction = self.xgboostreg._predict(X_last)[0]
            predictions.append(prediction)

            index_new = df_train.index[-1] + pd.Timedelta(1, unit='d')
            ar_df_index_new = self.ar_df[self.ar_df.index == index_new]

            if len(ar_df_index_new) == 0:
                res_num = 0
                res_vis_num = 0
            else:
                res_num = ar_df_index_new.reservation_num
                res_vis_num = ar_df_index_new.resv_visiters_num

            df_new_row = pd.DataFrame([[prediction,res_num,res_vis_num]], columns=df_train.columns, index=[index_new])
            df_train = pd.concat([df_train, df_new_row])

        predictions = self.xgboostreg._cut_under_zero(predictions)

        return predictions

def train_test_split(X, y, train_ratio):
    train_num = int(X.shape[0]*train_ratio)
    X_train = X[:train_num]
    y_train = y[:train_num]
    X_test = X[train_num:]
    y_test = y[train_num:]
    return X_train, X_test, y_train, y_test

def CV_gen(X, y, train_ratio=0.8):
    train_num = int(X.shape[0]*train_ratio)
    train = np.arange(0, train_num-1)
    test = np.arange(train_num-1, X.shape[0])

    yield train, test

def make_submission():
    data = datasets.RecruitDatasets()
    uih = uiu.UniqueIDHandler(data=data)
    unique_id_list = uih.get_sample_submission_unique_id_list()
    unique_id = unique_id_list[0]
    submission = su.SubmissionDataFrame()

    n, step = 30, 1

    # need to pass scorer for GridSearchCV.
    # http://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html
    ev_scorer = make_scorer(explained_variance_score)
    unique_id_num = len(unique_id_list)


    for i, unique_id in enumerate(unique_id_list):
        print(f"----- {i} / {unique_id_num} -----")
        params = {"n":n, "step":step, "unique_id":unique_id,\
                    "n_estimators":250, "subsample":0.5,\
                    "max_depth":30}

        uih = uiu.UniqueIDHandler(unique_id=unique_id, data=data)
        df = uih.get_dataframe()

        f_eng = fe.FeatureEngineearing(n, step, df)
        X, y = f_eng.get_X_y()

        # for submission
        xgboostreg_submission = XGBoostReg(**params)
        xgboostreg_submission = xgboostreg_submission.fit(X, y)
        pred_len = 39
        predictions = xgboostreg_submission.predict(X_test=np.arange(pred_len))

        predictions = submission.convert_submit_series(predictions)
        submission.concat(predictions, unique_id)

        del uih
        del df
        del f_eng
        del X, y
        del xgboostreg_submission, predictions

    submission.to_csv(savename="./submission/submission_xgboost_no_GSCV.csv")


if __name__ == '__main__':
    # import matplotlib.pyplot as plt

    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.plot(predictions, label="y_hat")
    # ax.plot(y_test, label="y")
    # fig.legend()
    # fig.show()

    # Grid Search
    # https://www.kaggle.com/phunter/xgboost-with-gridsearchcv
    data = datasets.RecruitDatasets()
    uih = uiu.UniqueIDHandler(data=data)
    unique_id_list = uih.get_sample_submission_unique_id_list()
    unique_id = unique_id_list[0]
    submission = su.SubmissionDataFrame()

    n, step = 30, 1

    # need to pass scorer for GridSearchCV.
    # http://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html
    ev_scorer = make_scorer(explained_variance_score)

    for unique_id in [unique_id_list[0]]:
        # These params need to be list.
        params = {"n":[n], "step":[step], "unique_id":[unique_id],\
                    "n_estimators":[250], "subsample":[0.5],\
                    "max_depth":[30]}

        uih = uiu.UniqueIDHandler(unique_id=unique_id, data=data)
        df = uih.get_dataframe()

        f_eng = fe.FeatureEngineearing(n, step, df)
        X, y = f_eng.get_X_y()

        xgboostreg = XGBoostReg()
        xgboostregGSCV = GridSearchCV(xgboostreg, params, cv=CV_gen(X, y, train_ratio=0.8), \
                                    iid=False, scoring=ev_scorer, verbose=2, n_jobs=-1)
        xgboostregGSCV.fit(X, y)
        print("unique_id : ", unique_id)
        print("params : ", xgboostregGSCV.best_params_)
        print("score : ", xgboostregGSCV.best_score_)

        # for submission
        xgboostreg_submission = XGBoostReg(**xgboostregGSCV.best_params_)
        xgboostreg_submission = xgboostreg_submission.fit(X, y)
        pred_len = 39
        predictions = xgboostreg_submission.predict(X_test=np.arange(pred_len))

        predictions = submission.convert_submit_series(predictions)
        submission.concat(predictions, unique_id)

        del uih
        del df
        del f_eng
        del X, y
        del xgboostreg, xgboostregGSCV
        del xgboostreg_submission, predictions

    submission.to_csv(savename="./submission/submission_xgboost_test.csv")


