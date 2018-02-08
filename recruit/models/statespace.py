import pandas as pd
from scipy import stats
import statsmodels.api as sm

from recruit.utils import datetime_utils as du

# for develop
from recruit import datasets
from recruit.utils import dataframe_utils as dfu
from recruit.utils import submission_utils as sub
from recruit.utils import unique_id_utils as uiu


# http://www.statsmodels.org/dev/generated/statsmodels.tsa.statespace.structural.UnobservedComponentsResults.html

# modify note
# Note that the test set intentionally spans a holiday week in Japan called the "Golden Week."

class StateSpaceModel(object):
    def __init__(self, df):
        self._init_endog(endog=df)
        self._init_model()

    def _init_endog(self, endog):

        if isinstance(endog, pd.core.series.Series):
            if not isinstance(endog.index[0], pd.tslib.Timestamp):
                # and index is not timestamp.
                raise KeyError("input dataframe should have visit_date column or whose index should be visit_date.")
            else:
                # index is timestamp
                self.endog = pd.DataFrame(endog)
        else:
            try:
                # in case that visit_date column exists.
                endog = endog[["visit_date", "visitors"]]
                self.endog = endog.set_index("visit_date")
            except KeyError:
                # in case that visit_date column does not exist.
                if not isinstance(endog.index[0], pd.tslib.Timestamp):
                    # and index is not timestamp.
                    raise KeyError("input dataframe should have visit_date column or whose index should be visit_date.")
                else:
                    # index is timestamp
                    self.endog = endog
            except AttributeError:
                raise AttributeError("input should be pandas dataframe, series.")

    # def _get_unique_id_from_subset(self, air_visit_subset):
    #     air_store_id = air_visit_subset.air_store_id.iloc[0]
    #     return air_store_id

    def _init_model(self):
        self.model = sm.tsa.UnobservedComponents(self.endog, 'local linear trend', seasonal=7)

    def fit(self, **kwargs):
        """
        arguments
            see http://www.statsmodels.org/dev/generated/statsmodels.tsa.statespace.structural.UnobservedComponents.fit.html#statsmodels.tsa.statespace.structural.UnobservedComponents.fit
        """
        res = self.model.fit(**kwargs)
        # print(res.summary()) for debug
        self.res = res
        return self

    def predict(self):
        # test_data_len = 38
        start_index = len(self.endog)
        pred_len = self._calc_gap_predlen()
        end_index = len(self.endog) + pred_len 

        # predict from train end to test end.
        pred = self.res.predict(start=start_index, end=end_index)

        # take subset from test start.
        pred = pred[pred.index >= pd.Timestamp("2017-04-23")]

        # substitute minus by 0.
        pred[pred < 0] = 0.
        self.pred = pred

        self.__check_pred_date()

        return self.pred

    def __check_pred_date(self):
        start = self.pred.index[0]
        end = self.pred.index[-1]

        assert start == pd.Timestamp("2017-04-23", freq="D"), f"start : {start}."
        assert end == pd.Timestamp("2017-05-31", freq="D"), f"end is {end}."

    def _calc_gap_predlen(self):
        """
        The gap is defined here as a range between train end and test start.
        The pred_len is defined here as a range between train end and test end.
        """

        test_start = pd.Timestamp("2017-04-23", freq="D")
        train_end = self.endog.index[-1]
        print("train end : ", train_end) # debug
        test_end = pd.Timestamp("2017-05-31", freq="D")
        gap = (test_start - train_end).days - 1
        print("gap : ", gap)

        day_delta = test_end - train_end
        print("day_delta : ", day_delta) # debug
        pred_len = day_delta.days - 1

        self.gap = gap # for debug
        return pred_len

def make_statespace_submit_ver1():
    """
        using only air_visiter data
        0.513
    """
    d = datasets.RecruitDatasets()
    avh = dfu.AirVisitHandler(d.air_visit)
    ssh = dfu.SampleSubmissionHandler(d.sample_sub)
    unique_id_list = ssh.get_unique_id()
    submission = sub.SubmissionDataFrame()

    for unique_id in unique_id_list:
        print("unique_id : ", unique_id)
        air_visit_subset = avh.get_df_subset(unique_id, completion=True)
        ssm = StateSpaceModel(air_visit_subset)
        ssm = ssm.fit()
        pred = ssm.predict()

        submission.concat(pred, unique_id)
        del ssm
        del air_visit_subset

    submission.to_csv(savename="./submission/submission_statespace_use_sampleid.csv")

def make_hoge():
    """
        memo
    """
    d = datasets.RecruitDatasets()
    uih = uiu.UniqueIDHandler(data=d)
    unique_id_list = uih.get_sample_submission_unique_id_list()
    unique_id = unique_id_list[0]
    df = uih.get_dataframe(unique_id)
    for i in range(len(df)):
        df.resv_visiters_num[i] = i
        df.reservation_num[i] = i**2

    # av_subset = uih.avh.get_df_subset(unique_id, completion=True)
    ssm = StateSpaceModel(df)
    ssm = ssm.fit(method='cg', cov_type='approx', optim_hessian="approx", cov_kwds={"approx_centered":True, "approx_complex_step":True})

import matplotlib.pyplot as plt
def check_pred():
    d = datasets.RecruitDatasets()
    avh = dfu.AirVisitHandler(d.air_visit)
    ssh = dfu.SampleSubmissionHandler(d.sample_sub)
    unique_id_list = ssh.get_unique_id()
    submission = sub.SubmissionDataFrame()

    unique_id = unique_id_list[0]
    print("unique_id : ", unique_id)
    air_visit_subset = avh.get_df_subset(unique_id, completion=True)

    train_num = int(air_visit_subset.shape[0] * 0.8)
    X_train = air_visit_subset.iloc[:train_num]
    y_train = air_visit_subset.visitors.values[:train_num]
    X_test = air_visit_subset.iloc[train_num:]
    y_test = air_visit_subset.visitors.values[train_num:]

    ssm = StateSpaceModel(X_train)
    ssm = ssm.fit()
    pred = ssm.predict()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(pred.values, label="y_hat")
    ax.plot(y_test, label="y")
    fig.legend()
    fig.show()

if __name__ == "__main__":
    pass

    # unique_id_list[0]
    # LinAlgError: Singular forecast error covariance matrix encountered at period $
    # https://github.com/statsmodels/statsmodels/blob/master/statsmodels/tsa/statespace/_statespace.pyx.in
