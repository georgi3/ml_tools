from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np


class ContextManagerChainedAssignments:
    """Context Manager for silencing pandas chained assignment warning"""
    def __init__(self, chained_assignment=None):
        acceptable = [None, 'warn', 'raise']
        if chained_assignment not in acceptable:
            raise ValueError(f'chain must be one of the following {acceptable}')
        self.chained_assignment = chained_assignment

    def __enter__(self):
        self.original_state = pd.options.mode.chained_assignment
        pd.options.mode.chained_assignment = self.chained_assignment
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pd.options.mode.chained_assignment = self.original_state


def ohe_leave_nan(dataframe, features, drop=None):
    """
    Purpose of this function is to be able to apply KNNImputer after encoding categorical features.
    Allows to pass pd.DataFrame to OneHotEncoder containing Nan values, drops Nan columns created by OHE and inserts Nan
    values at their indices at corresponding dummy columns. For example, if a feature contains 3 classes and contains Nan
    values at index 10, it will populate all three dummy columns with Nan values at index 10.
    :param dataframe: pd.DataFrame to be operated on
    :param features: list of column names to be transoformed
    :param drop: OneHotEncoder parameter, {'first', 'if_binary'}, default=None
                - do not pass 'if_binary' if binary columns have nan values, it will have no effect, pass 'first'
    :return: dataframe with encoded columns and nan values at place
    """
    ohe = OneHotEncoder(sparse=False, drop=drop)
    fitted = ohe.fit_transform(dataframe[features])  # fitting the dataframe with passed features
    columns = ohe.get_feature_names_out()  # getting column names back
    encoded_df = pd.DataFrame(fitted, columns=columns)  # creating encoded df
    nan_indices = {column[:-4]: encoded_df[encoded_df[column] == 1].index.tolist()
                   for column in columns if column.endswith('nan')}  # creating dict with nan indices for each column
    # ohe returns matrix with nan columns for each feature if nan were found in that column; finding & dropping such
    encoded_df.drop([col for col in encoded_df.columns.tolist() if col.endswith('nan')], axis=1, inplace=True)
    for feature, indices in nan_indices.items():
        # getting dummy columns that were created for each feature
        dummy_columns = [col for col in encoded_df.columns.tolist() if col.startswith(feature)]
        # inserting nan values back at their initial indices
        encoded_df.loc[indices, dummy_columns] = np.nan
    return encoded_df
