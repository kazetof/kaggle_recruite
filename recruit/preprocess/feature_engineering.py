import numpy as np

from recruit import datasets
from recruit.utils import unique_id_utils as uiu
from recruit.models import statespace as ss

class FeatureEngineearing(object):
    def __init__(self, n, step, unique_id_df):
        self.n = n
        self.step = step
        self.df = unique_id_df

    def get_X_y(self):
        X, y = self.get_df_lag()
        X_ssm = self.get_ssm_features()
        X = self.concat_features(X, X_ssm)
        return X, y

    def fieature_lag_generator(self, feature):
        feature_len = len(feature)
        for i in range(0, feature_len-self.n, self.step):
            feature_n_lag = feature[i:i+self.n]
            yield feature_n_lag

    def make_feature_lag_stack(self, feature):
        f_gen = self.fieature_lag_generator(feature=feature)
        new_row = next(f_gen)
        all_data = [new_row]
        for new_row in f_gen:
            all_data = np.vstack((all_data, new_row))
        return all_data

    def concat_features(self, feature1, feature2):
        f_concat = np.c_[feature1, feature2]
        return f_concat

    def get_ssm_features(self):
        ssm = ss.StateSpaceModel(self.df["visitors"])
        ssm = ssm.fit()

        level = ssm.res.level["filtered"]
        seasonal = ssm.res.seasonal["filtered"]
        trend = ssm.res.trend["filtered"]

        level_lag_lag = self.make_feature_lag_stack(level)
        seasonal_lag_lag = self.make_feature_lag_stack(seasonal)
        trend_lag_lag = self.make_feature_lag_stack(trend)

        # modify to concat by arbitrary num of features.
        X = self.concat_features(level_lag_lag, seasonal_lag_lag)
        X = self.concat_features(X, trend_lag_lag)
        return X

    def get_df_lag(self):
        uih = uiu.UniqueIDHandler()
        X, y = uih.get_lag_X_y(self.n, self.step, self.df)
        return X, y


if __name__ == '__main__':
    d = datasets.RecruitDatasets()
    uih = uiu.UniqueIDHandler(data=d)
    unique_id_list = uih.get_sample_submission_unique_id_list()
    unique_id = unique_id_list[0]
    df = uih.get_dataframe(unique_id)
    n, step = 30, 1
    f_eng = FeatureEngineearing(n, step, df)
    X, y = f_eng.get_X_y()





