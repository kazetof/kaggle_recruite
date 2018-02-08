import numpy as np

WEEKDAY = {0:"Monday", 1:"Tuesday", 2:"Wednesday", 3:"Thursday", 4:"Friday", 5:"Saturday", 6:"Sunday"}

def print_weekday(timestamp):
    """
    arguments
        timestamp : pandas.tslib.Timestamp
    """
    global WEEKDAY
    weekday_int = timestamp.weekday()
    print(f"{timestamp} : {WEEKDAY[weekday_int]}")

def get_no_data_weekday(air_visit_subset):
    """
    arguments
        air_visit_subset : pandas.core.frame.DataFrame
        air_visit_subset = avh.get_df_subset(unique_id[0])

        whole air_visit dataframe is allowed.
    """
    weekday_list = make_weekday_list(air_visit_subset)
    weekday_unique = np.unique(weekday_list)
    no_data_weekday = np.array(list(set(np.arange(7)) - set(weekday_unique)))
    return no_data_weekday

def weekdayint2str(weekday_list_int):
    """
    arguments
        weekday_list_int : numpy.ndarray
        output of make_weekday_list()
        Ex.
            In [64]: weekday_list_int
            Out[64]:
            array([2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 2, 3, 4, 5, 0, 1,

    returns 
        weekday_list_str : numpy.ndarray
            In [67]: weekday_list_str
            Out[67]:
            array(['Wednesday', 'Thursday', 'Friday', 'Saturday', 'Monday', 'Tuesday',
    """
    global WEEKDAY
    weekday_list_str = np.array([WEEKDAY[weekday_int] for weekday_int in weekday_list])
    return weekday_list_str

def make_weekday_list(air_visit):
    """
    arguments
        air_visit : pandas.core.frame.DataFrame
        The dtype of air_visit.visit_date should be datetime64[ns].

    returns
        weekday_list : numpy.ndarray
    """
    weekday_list = []
    for timestamp in air_visit.visit_date:
        weekday_int = timestamp.weekday()
        weekday_list.append(weekday_int)

    weekday_list = np.array(weekday_list)
    return weekday_list
