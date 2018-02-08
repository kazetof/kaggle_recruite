import numpy as np
import pandas as pd

class SubmissionDataFrame(object):
    """
    develop memo
        modify not to depend the prediction data type which is output of each model.
        now, it is dependent on output of statespace model by assuming pred is pd.Series whose index is timestamp.
    """
    def __init__(self):
        """
        arguments
            savefilename : str
                Ex. savefilename = "statespacemodel.csv"
        """
        self.df = self.init_logdf()

    def init_logdf(self):
        columns = ["id", "visitors"]
        df = pd.DataFrame(columns=columns)
        return df

    def append(self, new_row):
        """
        This method append one raw to self.df.

        arguments
            new_row : list like
                new_row vector whose shape is (n,) and n is len(columns).
        """
        new_row = np.array(new_row).reshape(1, len(self.columns))
        new_row_df = pd.DataFrame(new_row, columns=self.columns)
        self.df = pd.concat([self.df, new_row_df], ignore_index=True)
        return self

    def _make_submission_id(self, unique_id, timestamp):
        submission_id = unique_id + "_" + timestamp.strftime(format="%Y-%m-%d")
        return submission_id

    def concat(self, pred, unique_id):
        # is it better that argument is my defined model object?
        submission_df = self._make_submission_df(pred, unique_id)
        self.df = pd.concat([self.df, submission_df], ignore_index=True)
        return self

    def _make_submission_id_col(self, pred, unique_id):
        """
        arguments
            pred : pandas.core.series.Series
                Ex.
                    In [61]: pred
                    Out[61]:
                    2017-04-23    -0.696218
                    2017-04-24     7.271829
                The index is datetime whose dtype is dtype='datetime64[ns]'.

        returns
            submission_id_col : pandas.core.series.Series
            Name: id, dtype: object
                Ex.
                    In [81]: submission_id_col
                    Out[81]:
                    0     air_ba937bf13d40fb24_2017-04-23
                    1     air_ba937bf13d40fb24_2017-04-24
                    2     air_ba937bf13d40fb24_2017-04-25
        """
        submission_id_col = pd.Series([self._make_submission_id(unique_id, timestamp) for timestamp in pred.index], name="id")
        return submission_id_col

    def _make_submission_df(self, pred, unique_id):
        submission_id_col = self._make_submission_id_col(pred, unique_id)
        data = {"id":submission_id_col.values, "visitors":pred.values}
        submission_df = pd.DataFrame(data)
        return submission_df

    def to_csv(self, savename=None):
        if savename is None:
            savename = "./submission/submission.csv"
        self.df.to_csv(savename, index=False)

    @staticmethod
    def convert_submit_series(predictions):
        """
           prediction : ndarray
                shape is (39,).
        """
        if len(predictions) != 39:
            raise ValueError("The predictions length should be 39.")

        index = pd.date_range(start="2017-04-23", end="2017-05-31")
        predictions = pd.Series(predictions, index=index)
        return predictions


# Evaluation Exception: Submission must have 32019 rows




            