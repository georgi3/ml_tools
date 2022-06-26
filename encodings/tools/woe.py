from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np


# TODO: add report method (features with their respective information value score)
class WoEncoder(BaseEstimator, TransformerMixin):
    """
    Column Transformer using Scikit-Learn interface. Weight of Evidence transformer which is compatible with
    Scikit-Learn pipeline. It could handle both continuous and categorical variables but not at the same time.
    Transformer creates WoE a scheme for a dataframe:
    e.g.
        {
        'continuous_scheme':
            {
            'feature_1':
                {
                'group_1': {
                    'members': {
                        'bin_1': 0.23,
                        'bin_2': 0.25
                                },
                    'woe_mean': 0.24
                },
                'group_2': {
                    'members': {
                        'bin_3': 0.035,
                        'bin_4': 0.036
                                },
                    'woe_mean': 0.037
                    }
                },
            'feature_2':{...}
            },
        'categorical_scheme':
            {
            ...
            }
        }

    :param is_continuous: bool, True if features that are passed continuous, False if categorical
    :param smoothing_alpha: float, minimum percent of data points each bin should represent, defaults to 0.05 (%5).
                            Applied only on continuous features
    :param coarse_alpha: float, threshold within which WoE values for neighbouring bins should be coarsed
    :param drop_insufficient_iv: bool, drop features with insufficient Information Value
    :param iv_threshold: float, threshold of Information Value for feature to be considered sufficient
    """

    def __init__(self, is_continuous=True, drop_insufficient_iv=True, smoothing_alpha=0.05, coarse_alpha=0.05,
                 iv_threshold=0.02):

        self.is_continuous = is_continuous
        self.drop_insufficient_iv = drop_insufficient_iv
        self.smoothing_alpha = smoothing_alpha
        self.coarse_alpha = coarse_alpha
        self.iv_threshold = iv_threshold

    def get_optimal_number_of_bins(self, dataframe, feature: str):
        """
        Calculates optimal number bins based on the threshold, where threshold is self.alpha
        :param dataframe: pandas.Series
        :param feature: str, continuous feature of the DataFrame
        :return: int
        """
        total = dataframe[feature].count()                  # total number of data points
        min_entries = int(total * self.smoothing_alpha)     # min number of data points per bin
        optimal = int(total / min_entries)                  # optimal number of bins
        df = pd.DataFrame(columns=[feature])
        for q in range(optimal, 1, -1):                     # trying max number of bin, if passes threshold, returns
            df[feature] = dataframe[feature]                # else decrements q by one
            try:
                df[feature] = pd.qcut(df[feature], q=q)
            except TypeError as err:
                raise TypeError(f'{err}.\n CUSTOM: Possible reason: not enough number of unique values in the feature'
                                f' to represent distribution proportionally.')
            if min_entries <= min(df[feature].value_counts()):
                return q

    @staticmethod
    def bin_feature(dataframe, feature: str, q: int):
        """
        Transforms continuous features to bins.
        :param dataframe: pandas.DataFrame
        :param feature: str, continuous feature
        :param q: int, number of quantiles
        :return: pandas.DataFrame
        """
        dataframe[feature] = pd.qcut(dataframe[feature], q=q)
        # needed because names will be used as strings (dtype is pandas.interval_range if not casted)
        dataframe[feature] = dataframe[feature].astype(str)
        return dataframe

    @staticmethod
    def calculate_woe_iv(dataframe, feature: str):
        """
        Calculates Weight of Evidence and Information Value for passed categorical feature.
        :param dataframe: pandas.DataFrame
        :param feature: str, categorical feature
        :return: tuple, WoE DataFrame and IV for the feature
        """
        stats = []
        bins_n = dataframe[feature].nunique()                       # number of bins in the feature
        for i in range(bins_n):
            bin_name = dataframe[feature].unique()[i]
            stats.append({
                'bin_name': bin_name,
                'bin_total_count': (dataframe[feature] == bin_name).sum(),
                'bin_pos_count': ((dataframe[feature] == bin_name) & (dataframe['target'] == 1)).sum(),
                'bin_neg_count': ((dataframe[feature] == bin_name) & (dataframe['target'] == 0)).sum()
            })
        woe_df = pd.DataFrame(stats)
        woe_df['distr_pos'] = woe_df['bin_pos_count'] / woe_df['bin_pos_count'].sum()
        woe_df['distr_neg'] = woe_df['bin_neg_count'] / woe_df['bin_neg_count'].sum()

        np.seterr(divide='ignore')                                  # lifting-up the warning because it is handled below
        woe_df['WoE'] = np.log(woe_df['distr_pos'] / woe_df['distr_neg'])
        np.seterr(divide='warn')                                    # setting the warning back

        woe_df = woe_df.replace({'WoE': {np.inf: 0, -np.inf: 0}})
        woe_df['IV'] = (woe_df['distr_pos'] - woe_df['distr_neg']) * woe_df['WoE']
        woe_df.sort_values(by='WoE', inplace=True)
        iv = woe_df['IV'].sum()
        return woe_df, iv

    def iv_is_sufficient(self, iv):
        """
        Checks if information value score above the threshold (0.02)
        Information Value Rule of Thumb Table:

                <0.02	Not useful for prediction
                0.02 to 0.1	Weak predictive Power
                0.1 to 0.3	Medium predictive Power
                0.3 to 0.5	Strong predictive Power
                >0.5	Suspicious Predictive Power

        :param iv: float,  Information Value score
        :return: bool
        """
        threshold = self.iv_threshold
        if iv > threshold:
            return True
        else:
            return False

    def more_to_coarse(self, coarsed_dict):
        """
        Checks if coarsed neighbouring (in ascending order) bins are within the threshold of coarsing
        :param coarsed_dict: dict,  dictionary of coarsed bins and their mean values.
                            Keys are concatenated bin names, values are their mean weight of evidence.
        :return: bool
        """
        result = False
        coarsed_lst = [(k, v) for k, v in coarsed_dict.items()]         # turning dict to tuple for operating purposes
        to_coarse = sorted(coarsed_lst, key=lambda item: item[1])       # double-checking ascending order
        for index in range(len(to_coarse) - 1):
            this_value = to_coarse[index][1]
            other_value = to_coarse[index + 1][1]
            if abs(this_value - other_value) < self.coarse_alpha:
                result = True
        return result

    def coarse_grouper(self, woes_dict):
        """
        Recursive function, that receives dictionary of bins and their weight of evidences. Sorts them in ascending
        order. Checks in pairs if neighbouring bins have delta of their WoE withing threshold (where threshold is
        self.coarse_alpha). After iterating once saves dict of coarsed bins and their mean WoEs. Passes this dict to
        more_to_coarse(), if True calls itself again, else returns list of lists (where each list is a group of bins
        to be coarsed).

        Note:
        This algorithm has its flaws. For example, if there are three neighbours with values of (0.05, 0.1, 0.125),
        first and third elements should not be coarsed because their delta is 0.075. However, the algorithm will coarse
        first element with the second first. On the next iteration it'll compare first's and second's element mean
        value (0.075) to (0.125), and their delta will be (0.05), so algorithm will decide to coarse them as well.
        Therefore, threshold is not strictly what it is when set is initialized.

        Parameters:
        :param woes_dict: dict
        :return: list (of lists)
        """
        woes_tuples = sorted([(k, v) for k, v in woes_dict.items()], key=lambda item: item[1])
        coarsed_woes = {}
        iterated_over = set()

        for value in woes_tuples:
            this_name = value[0]
            this_value = value[1]
            next_index = woes_tuples.index(value) + 1       # getting index of the next element to get its neighbour
            if next_index < len(woes_tuples):               # check to prevent IndexError
                other_name = woes_tuples[next_index][0]     # next ascending neighbour
                other_value = woes_tuples[next_index][1]
            else:                                           # if check fails, break out of the loop
                if this_name in iterated_over:              # before breaking, check if last element was written down
                    break                                   # if True: break; else write it down to dictionary, used if
                else:                                                           # list has odd number of elements
                    coarsed_woes[this_name] = this_value
                    break

            if (this_name in iterated_over) or (other_name in iterated_over):   # skipping previous neighbour, to avoid
                continue                                                        # repetition. i.e. next pair
            elif abs(this_value - other_value) < self.coarse_alpha:
                name = str(this_name) + '&' + str(other_name)
                coarsed_woes[name] = (this_value + other_value) / 2             # mean woe
                iterated_over.add(this_name)
                iterated_over.add(other_name)
            else:
                coarsed_woes[this_name] = this_value
                iterated_over.add(this_name)

        groups_to_coarse = [key.split('&') for key in coarsed_woes]             # list of lists to be returned
        if self.more_to_coarse(coarsed_woes):                                   # if more values to coarse, call itself
            return self.coarse_grouper(coarsed_woes)
        else:
            return groups_to_coarse

    @staticmethod
    def get_woe_feature_scheme(groups, woe_df):
        """
        Creates a scheme for a single feature, example see at the top
        :param groups: list (of lists)
        :param woe_df: pandas.DataFrame
        :return: dict
        """
        feature_scheme = dict()
        for group, i in zip(groups, list(range(len(groups)))):
            group_name = 'group_' + str(i)
            feature_scheme[group_name] = {}
            feature_scheme[group_name]['members'] = {}
            feature_scheme[group_name]['group_woe_mean'] = woe_df[woe_df['bin_name'].isin(group)]['WoE'].mean()
            for member in group:
                feature_scheme[group_name]['members'][member] = woe_df[woe_df['bin_name'] == member]['WoE'].item()
        return feature_scheme

    def analyze_continuous(self, dataframe):
        """
        Pipeline for continuous features.
        :param dataframe: pandas.DataFrame
        :return: dict (scheme for continuous features)
        """
        continuous_woes_scheme = dict()
        for feature in dataframe:
            if feature == 'target':
                continue
            q = self.get_optimal_number_of_bins(dataframe, feature)             # optimal number of bins
            df = self.bin_feature(dataframe, feature, q)                        # transforming continuous feature
            woe, iv = self.calculate_woe_iv(df, feature)                        # calculating WoE, IV for feature
            if self.iv_is_sufficient(iv):                                       # check for IV sufficiency
                woes_dict = {k: v for k, v in zip(woe['bin_name'].tolist(), woe['WoE'].tolist())}
                groups = self.coarse_grouper(woes_dict)                         # getting groups for bins to be coarsed
                feature_scheme = self.get_woe_feature_scheme(groups, woe)       # scheme for the feature
                continuous_woes_scheme[feature] = feature_scheme
            else:
                self.insufficient_features.append(feature)
        return continuous_woes_scheme

    def analyze_categorical(self, df):
        """
        Pipeline for categorical features.
        :param df: pandas.DataFrame
        :return: dict (scheme for categorical features)
        """
        categorical_woes_scheme = dict()
        for feature in df:
            if feature == 'target':
                continue
            woe, iv = self.calculate_woe_iv(df, feature)                    # calculating WoE, IV for feature
            if self.iv_is_sufficient(iv):                                   # check for IV sufficiency
                woes_dict = {k: v for k, v in zip(woe['bin_name'].tolist(), woe['WoE'].tolist())}
                groups = self.coarse_grouper(woes_dict)                     # getting groups for bins to be coarsed
                feature_scheme = self.get_woe_feature_scheme(groups, woe)   # scheme for the feature
                categorical_woes_scheme[feature] = feature_scheme
            else:
                self.insufficient_features.append(feature)
        return categorical_woes_scheme

    def drop_insufficient_features(self, X: pd.DataFrame):
        """
        If drop_insufficient_iv=True, drops such columns.
        :param X: pd.DataFrame
        :return: pd.DataFrame
        """
        if self.drop_insufficient_iv:
            X.drop(self.insufficient_features, axis=1, inplace=True)
        else:
            print(f'{self.insufficient_features} are not dropped!')

    def fit(self, X: pd.DataFrame, y=None):
        """
        Creates a scheme for passed variables, Scikit-Learn interface
        :param X: pd.Dataframe, independent variables
        :param y: pd.Series, dependent variables
        :return: dict, scheme
        """
        dataframe = X.copy(deep=True)
        self.scheme = dict()
        self.insufficient_features = []
        if self.is_continuous:
            dataframe['target'] = y
            cont_scheme = self.analyze_continuous(dataframe)
            self.scheme['continuous_scheme'] = cont_scheme
            self.interpreted_scheme, self.bin_scheme = self.interpret_scheme()
        else:
            dataframe = dataframe.astype(str)
            dataframe['target'] = y
            cat_scheme = self.analyze_categorical(dataframe)
            self.scheme['categorical_scheme'] = cat_scheme
            self.interpreted_scheme = self.interpret_scheme()
        return self

    def transform(self, X, y=None):
        """
        Transforms dataframe based on WoE encoding. Scikit-Learn interface.
        :param X: pd.DataFrame, independent variables
        :param y: pd.Series, target variable
        :return: pd.DataFrame
        """
        X = pd.DataFrame(X)
        self.drop_insufficient_features(X)
        X = self.bin_dataframe(X)
        X = X.replace(self.interpreted_scheme)
        return X

    def bin_dataframe(self, X):
        """
        If dataframe is continuous, features will be binned when called transformed. Uses interpreted scheme as the map.
        :param X: pd.DataFrame, independent variables
        :return: pd.DataFrame
        """
        if self.is_continuous:
            X = X.copy(deep=True)
            for feature, bins in self.bin_scheme.items():
                X.loc[:, feature] = pd.cut(X[feature], bins=bins)
            return X
        return X

    @staticmethod
    def creating_bins(feature_scheme):
        """
        Creates bins for pandas.cut()
        :param feature_scheme: dict
        :return: list, of tuples with pd.Intervals
        """
        tuples = [(interval.left, interval.right) for interval in feature_scheme]
        set_ = set()
        for item in tuples:
            set_.add(item[0])
            set_.add(item[1])
        bins = sorted(set_)
        return bins

    def create_bin_scheme(self, adjusted_scheme):
        """
        Creates bin schemes for continuous features
        :param adjusted_scheme:
        :return: dict
        """
        bin_scheme = dict()
        for feature in adjusted_scheme:
            bins = self.creating_bins(adjusted_scheme[feature])
            bin_scheme[feature] = bins
        return bin_scheme

    @staticmethod
    def adjust_bin_edges(cont_scheme):
        """
        Adjusts the most left and right boundaries to inf, -inf because training data might not have the most extreme
        data point
        :param cont_scheme: dict, continuous scheme
        :return dict
        """
        adjusted_edges = dict()
        for feature in cont_scheme:
            asc_lst_tuples = sorted(cont_scheme[feature].items(), key=lambda interval: interval[0].left)
            asc_lst = [[interval, value] for interval, value in asc_lst_tuples]
            asc_lst[0][0] = pd.Interval(left=-np.inf, right=asc_lst[0][0].right)
            asc_lst[-1][0] = pd.Interval(left=asc_lst[-1][0].left, right=np.inf)
            asc_dict = dict(asc_lst)
            adjusted_edges[feature] = asc_dict
        return adjusted_edges

    @staticmethod
    def continuous_scheme_interpreter(converted_scheme):
        """
        Converts keys to pandas.Interval for continuous scheme
        :param converted_scheme: dict, interpreted scheme
        :return: dict, with keys as pandas.Intervals and values as WoE
        """
        interpreted = dict()
        for feature, feature_info in converted_scheme.items():
            interpreted[feature] = {}
            for key, value in feature_info.items():
                start, end = map(lambda interval: float(interval), key[1:-1].split(','))
                new_key = pd.Interval(start, end)
                interpreted[feature][new_key] = value
        return interpreted

    def scheme_reader(self):
        """
        Scheme reader
        :return: dict
        """
        interpreted = dict()
        scheme_type = 'continuous_scheme' if self.is_continuous else 'categorical_scheme'
        for feature_name, feature_info in self.scheme[scheme_type].items():
            interpreted[feature_name] = {}
            for group_name, group_info in feature_info.items():
                for member in group_info.get('members'):
                    interpreted[feature_name][member] = group_info.get('group_woe_mean', np.nan)
        return interpreted

    def interpret_scheme(self):
        """
        Interprets WoE scheme
        :return: dict
        """
        converted_scheme = self.scheme_reader()
        if self.is_continuous:
            continuous_scheme = self.continuous_scheme_interpreter(converted_scheme)
            adjusted_continuous_scheme = self.adjust_bin_edges(continuous_scheme)
            return adjusted_continuous_scheme, self.create_bin_scheme(adjusted_continuous_scheme)
        else:
            return converted_scheme
