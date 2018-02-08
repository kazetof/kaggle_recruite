import pandas as pd
import numpy as np

from recruit.utils import dataframe_utils as dfu
from recruit import datasets

class UniqueIDHandler(object):
    """
        this class handle all dataframe like air_visit, air_reserve for each unique id.

        Ex.
            d = datasets.RecruitDatasets()
            uih = uiu.UniqueIDHandler(data=d)
            unique_id_list = uih.get_sample_submission_unique_id_list()
            df = uih.get_dataframe(unique_id_list[0])
    """
    def __init__(self, unique_id=None, data=None):
        if data is None:
            self.data = datasets.RecruitDatasets()
        else:
            self.data = data
        
        self.ssh = dfu.SampleSubmissionHandler(self.data.sample_sub)

        if unique_id is not None:
            self.set_unique_id(unique_id)

    def set_unique_id(self, unique_id):
        self.unique_id = unique_id
        self.arh = dfu.AirReserveHandler(self.data.air_reserve)
        self.avh = dfu.AirVisitHandler(self.data.air_visit)

        return self

    def _get_resnum_resvisnum_df(self):
        """
        returns
            output_df : pandas.core.frame.DataFrame
                a dataframe whose visit_date is completed.
                In [99]: output_df
                Out[99]:
                    visit_date  reservation_num  visitors_num
                0   2016-01-02              0.0           0.0
                1   2016-01-03              1.0           6.0

        """
        df_subset = self.arh.get_df_subset(self.unique_id)
        visiters_num = self.arh.get_visitdate_visiters_num_df(df_subset)
        visiters_num = self._fit_datetime_within_train_range(visiters_num)
        visiters_num = visiters_num.reset_index()
        visiters_num = visiters_num.rename(columns={"index": "visit_date"})

        # complete visiters_num by air_visit's visit datetime range.
        completed_date = self.avh.get_completed_date(self.unique_id)
        completed_date_df = pd.DataFrame(completed_date)

        output_df = visiters_num.merge(completed_date_df, left_on="visit_date", right_on="visit_date", how='right')
        output_df = output_df.fillna(0)
        output_df = output_df.sort_values(by="visit_date").reset_index(drop=True)

        return output_df

    def get_dataframe(self, unique_id=None):
        """
        returns
            output_df : pandas.core.frame.DataFrame
            Ex.
                In [9]: output_df
                Out[9]:
                            visitors  reservation_num  resv_visiters_num
                visit_date
                2016-07-01      35.0              0.0                0.0
                2016-07-02       9.0              0.0                0.0
        """
        if unique_id is not None:
            self.set_unique_id(unique_id)

        try:
            self.unique_id
        except AttributeError:
            raise AttributeError("do set_unique_id() before get_dataframe() or pass unique_id.")

        air_visit_subset = self.avh.get_df_subset(self.unique_id, completion=True)
        air_reserve_subset = self._get_resnum_resvisnum_df()

        output_df = air_visit_subset.merge(air_reserve_subset, left_on="visit_date", right_on="visit_date", how='left')
        output_df = output_df.drop("air_store_id", axis=1)
        output_df = output_df.set_index("visit_date")
        return output_df


    def _fit_datetime_within_train_range(self, visiters_num):
        """
        datetime in air_reservation could be over the train data range like below.
            74 2017-04-22                3            19
            75 2017-04-23                1             6
            76 2017-04-24                1             2
            77 2017-04-28                1            10
            78 2017-05-03                1             6
        so cut the last parts to fit range.
        Can I use these exceeded values?
        """
        end = pd.Timestamp("2017-4-22")
        visiters_num_fitted = visiters_num[visiters_num.index <= end]
        return visiters_num_fitted

    def get_sample_submission_unique_id_list(self):
        """
        returns
            unique_id_list : numpy.ndarray
            Ex.
                In [4]: unique_id_list
                Out[4]:
                array(['air_00a91d42b08b08d9', 'air_0164b9927d20bcc3',
                       'air_0241aa3964b7f861', 'air_0328696196e46f18',

        """
        unique_id_list = self.ssh.get_unique_id()
        return unique_id_list

    def _n_lag_generator(self, df, n, step=1):
        """
            df : pandas.core.frame.DataFrame
                pandas.core.series.Series is allowed.
                Ex.
                    In [196]: df
                    Out[196]:
                                visitors  reservation_num  resv_visiters_num
                    visit_date
                    2016-07-01      35.0              0.0                0.0
                    2016-07-02       9.0              0.0                0.0
            n : int
            step : int
        """
        df_len = len(df)

        for i in range(0, df_len-n, step):
            df_n_lag = df.iloc[i:i+n]

            y = df.visitors.iloc[i+n]
            yield df_n_lag, y

    def get_lag_X_y(self, n, step, df=None):
        if df is None:
            df = self.get_dataframe()

        n_lag_gen = self._n_lag_generator(df, n, step)

        df_subset, y = next(n_lag_gen)
        all_X = df_subset.values.T.flatten()
        all_y = [y]

        for df_subset, y in n_lag_gen:
            new_row = df_subset.values.T.flatten()
            all_X = np.vstack((all_X,new_row))
            all_y.append(y)
            del new_row

        all_y = np.array(all_y)
        return all_X, all_y






