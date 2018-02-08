import pandas as pd
import numpy as np
import re

from recruit.utils import datetime_utils as du

class DataFrameHandlerBase(object):
    """
        base class
    """
    def __init__(self, df):
        self.df = df

    def get_unique_id(self):
        unique_id_list = self.df.air_store_id.unique()
        return unique_id_list

    def to_datetime(self, date_colname):
        """
        Ex.
            d = Datasets.RecruitDatasets()
            avh = dfu.AirVisitHandler()
            d.air_visit = avh.to_datetime(d.air_visit)
        """
        self.df[date_colname] = pd.to_datetime(self.df[date_colname])

    def get_df_subset(self, unique_id):
        """
        arguments
            unique_id : str
                Ex. unique_id = "air_8e4360a64dbd4c50"
            completion : bool
                if True, the subset dataframe will be completed by visit_date filled by 0 in visitors.

        returns
            air_visit_subset : pandas.core.frame.DataFrame
        """
        df_subset = self.df[self.df.air_store_id == unique_id]

        # if completion:
        #     df_subset = self._df_completion_by_datetime(df_subset)

        return df_subset

    def _datetime_completion(self, datetime_list):
        """

        visit_date : pandas.core.series.Series
            The column of air_visit_subset

        return
            timestamp_list : list
            Ex. 
                In [138]: timestamp_list
                Out[138]:
                [Timestamp('2016-07-01 00:00:00'),
                 Timestamp('2016-07-02 00:00:00'),
                 Timestamp('2016-07-03 00:00:00'),

        Ex. timestamp_list = datetime_completion(air_visit_subset.visit_date)
        """
        start_date = datetime_list.iloc[0]
        end_date = datetime_list.iloc[-1]

        timestamp_list = [start_date]
        time_delta = pd.Timedelta(1, unit="D")

        for i in np.arange(len(datetime_list)-1):
            t = datetime_list.iloc[i]
            t_1_data = datetime_list.iloc[i+1]
            t_1 = t + time_delta

            timestamp_list.append(t_1)

            while t_1_data != t_1:
                t_1 = t_1 + time_delta
                timestamp_list.append(t_1)

                # print(f"{t_1} has completed.") # for debug

        return timestamp_list

class SampleSubmissionHandler(DataFrameHandlerBase):
    def __init__(self, sample_submission):
        self.df = sample_submission

    def get_unique_id(self):
        pattern = r"_[0-9]{4}-[0-9]{2}-[0-9]{2}"
        repl = ""
        unique_id_list = np.unique([ re.sub(pattern, repl, string) for string in self.df.id ])

        return unique_id_list

class AirReserveHandler(DataFrameHandlerBase):
    def __init__(self, air_reserve):
        self.df = air_reserve
        self.to_datetime("visit_datetime")
        self.to_datetime("reserve_datetime")

    def make_reserve_df(self):
        """
            making dataframe whose columns are datetime and reserve_visit_num.
            no need?
        """
        pass

    def _extract_date(self, visit_datetime):
        """
        extract only date in 2016-01-01 19:00:00 then convert it to str like 2016-01-01.

        arguments
            visit_datetime : pandas.core.series.Series
            Ex.
                In [61]: visit_datetime
                Out[61]:
                0       2016-01-01 19:00:00
                3       2016-01-01 20:00:00
                ...
                Name: visit_datetime, dtype: datetime64[ns]

        returns
            visit_date : pandas.core.series.Series
            Ex.
                In [63]: visit_date
                Out[63]:
                0       2016-01-01
                1       2016-01-01
                ...
                Name: visit_datetime, dtype: object
        """
        visit_date = pd.Series([str(datetime).split(" ")[0] for datetime in visit_datetime], name="visit_datetime")
        return visit_date

    def get_visitdate_visiters_num_df(self, df_subset):
        """
        arguments
            df_subset : pandas.core.frame.DataFrame
            Ex. 
                df_subset = self.get_df_subset(unique_id[0])

                In [69]: df_subset
                Out[69]:
                               air_store_id      visit_datetime    reserve_datetime  \
                0      air_877f79706adbfb06 2016-01-01 19:00:00 2016-01-01 16:00:00
                3      air_877f79706adbfb06 2016-01-01 20:00:00 2016-01-01 16:00:00

        returns
            visitdate_visiters_num_df : pandas.core.frame.DataFrame
            Ex.
                In [2]: visitdate_visiters_num_df
                Out[2]:
                            visit_date_num  visitors_num
                2016-07-02               1             0
                2016-07-08               2             0

        code example
            from tests.test_dataframe_utils import TestAirReserveHandler
            test = TestAirReserveHandler()
            test.setUpClass()
            unique_id = test.arh.get_unique_id()
            df_subset = test.arh.get_df_subset(unique_id[0])
            visit_date_num = test.arh.count_reserve_visit(df_subset)
        """
        # make visit_date_num column
        visit_date = self._extract_date(df_subset.visit_datetime)
        visit_date_num = visit_date.value_counts()
        visit_date_num.index = pd.to_datetime(visit_date_num.index)
        visit_date_num = visit_date_num.sort_index()

        # make visitors_num column
        visitors_num_list = []
        for unique_date in visit_date_num.index:
            unique_date_str = self._extract_date([unique_date])[0]
            visitors_num = df_subset.reserve_visitors.values[visit_date == unique_date_str].sum()
            visitors_num_list.append(visitors_num)

        visitors_num_list = pd.Series(visitors_num_list, name="visitors_num", index=visit_date_num.index)

        data = {"reservation_num" : visit_date_num, "resv_visiters_num" : visitors_num_list}
        visitdate_visiters_num_df = pd.DataFrame(data)
        return visitdate_visiters_num_df

    def complesion(self):
        pass
        # input is AirVisit subset
        # and complete date time by 0
        # if reservation is 0, tile 0.


class AirVisitHandler(DataFrameHandlerBase):
    def __init__(self, air_visit):
        self.df = air_visit
        self.to_datetime(date_colname="visit_date")

    def get_unique_id(self):
        """
        It looks like using unique_id_list in sample submission is good, 
        so this method is lower priority.
        """
        air_visit_unique_id = self.df.air_store_id.unique()
        return air_visit_unique_id

    def get_df_subset(self, unique_id, completion=False):
        """
        arguments
            unique_id : str
                Ex. unique_id = "air_8e4360a64dbd4c50"
            completion : bool
                if True, the subset dataframe will be completed by visit_date filled by 0 in visitors.

        returns
            air_visit_subset : pandas.core.frame.DataFrame
        """
        air_visit_subset = self.df[self.df.air_store_id == unique_id]

        if completion:
            air_visit_subset = self._df_completion_by_datetime(air_visit_subset)

        return air_visit_subset

    def check_no_data_weekday(self):
        """
        just print no data weekday int.
        """
        unique_id_list = self.get_unique_id()

        for unique_id in unique_id_list:
            air_visit_subset = self.get_df_subset(unique_id)
            no_data_weekday = du.get_no_data_weekday(air_visit_subset)
            print(no_data_weekday)


    # should I move it to base class?
    def _df_completion_by_datetime(self, air_visit_subset):
        """
        arguments
            air_visit_subset : pandas.core.frame.DataFrame
            Ex.
                In [133]: air_visit_subset
                Out[133]:
                              air_store_id visit_date  visitors
                711   air_8e4360a64dbd4c50 2016-07-01        28

        returns
            df_new : pandas.core.frame.DataFrame
                This dataframe is completed the skipped date in air_visit_subset and whose visitors value is 0.
        """
        timestamp_list = self._datetime_completion(air_visit_subset.visit_date)
        completed_visit_date = pd.DataFrame(timestamp_list, columns=["visit_date"])

        df_new = pd.merge(air_visit_subset, completed_visit_date, on='visit_date', how='outer')
        df_new.visitors = df_new.visitors.fillna(0)
        unique_id = air_visit_subset.air_store_id.iloc[0] # it assume that air_store_id are all same value.
        df_new.air_store_id = df_new.air_store_id.fillna(unique_id)
        df_new = df_new.sort_values(by="visit_date")

        return df_new

    def _datetime_completion(self, datetime_list):
        """
            returns
                timestamp_list : list
        """
        timestamp_list = super()._datetime_completion(datetime_list)
        # write a code here to add timestamp until reaching 4/22
        end = pd.Timestamp("2017-04-22")
        time_delta = pd.Timedelta(1, unit="D")

        t = timestamp_list[-1]
        while t != end:
            t_1 = t + time_delta
            timestamp_list.append(t_1)
            t = timestamp_list[-1]

        return timestamp_list

    def get_completed_date(self, unique_id):
        df_subset = self.get_df_subset(unique_id, completion=True)
        completed_date = df_subset["visit_date"]
        return completed_date



