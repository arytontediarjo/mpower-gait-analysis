"""

Author: Sage Bionetworks

Script to group features by healthcodes,
with several different aggregation

"""

# import future libraries
from __future__ import print_function
from __future__ import unicode_literals

# import standard libraries
import time
import pandas as pd
import numpy as np


# import external libraries
import synapseclient as sc

# import project modules
from utils import query_utils as query


# global variables
data_dict = {
    "FEATURE_DATA": {
        "MPOWER_V1": "syn21597373",
        "MPOWER_V2": "syn21597625",
        "MPOWER_PASSIVE": "syn21597842",
        "ELEVATE_MS": "syn21597862"},
    "METADATA": {
        "ACTIVE_WALKING": "syn21597317",
        "ACTIVE_BALANCE": "syn21599329",
        "PASSIVE": "syn21597318"},
    "OUTPUT_INFO": {
        "PARENT_FOLDER": "syn21592268",
        "PROJ_REPO_NAME": "mpower-gait-analysis",
        "PATH_GITHUB_TOKEN": "~/git_token.txt"}
}
syn = sc.login()


def iqr(x):
    """
    Function for getting IQR value
    """
    return q75(x) - q25(x)


def q25(x):
    """
    Function for getting first quantile
    """
    return x.quantile(0.25)


def q75(x):
    """
    Function for getting third quantile
    """
    return x.quantile(0.75)


def valrange(x):
    """
    Function for getting the value range
    """
    return x.max() - x.min()


def kurtosis(x):
    """
    Function to retrieve kurtosis
    """
    return x.kurt()


def skew(x):
    """
    Function to retrieve skewness
    """
    return x.skew()


def groupby_wrapper(data, group, exclude_columns=[]):
    """
    Wrapper function to wrap feature data
    into several aggregation function

    Args:
        data (dtype: pd.Dataframe)   : feature datasets
        group (dtype: string)        : which group to aggregate
        exclude_columns (dtype: list): columns to exclude during groupby
    Returns:
        Rtype: pd.Dataframe
        Returns grouped healthcodes features
    """

    features_cols = [feat for feat in data.columns if
                     feat not in exclude_columns]
    data = data[features_cols].groupby(group).agg([np.max,
                                                   np.median,
                                                   np.mean,
                                                   q25, q75,
                                                   valrange, iqr])
    feature_cols = []
    for feat, agg in data.columns:
        feature_cols_name = "{}_{}".format(agg, feat)
        feature_cols.append(feature_cols_name)
    data.columns = feature_cols
    return data


def main():

    used_script_url = query.get_git_used_script_url(
        path_to_github_token=data_dict["OUTPUT_INFO"]["PATH_GITHUB_TOKEN"],
        proj_repo_name=data_dict["OUTPUT_INFO"]["PROJ_REPO_NAME"],
        script_name=__file__)

    metadata_cols = ['appVersion', 'createdOn',
                     'phoneInfo', 'recordId',
                     'table_version', 'test_type']

    for key, feature_synId in data_dict["FEATURE_DATA"].items():
        if "PASSIVE" in key:
            metadata_synId = data_dict["METADATA"]["PASSIVE"]
        else:
            metadata_synId = data_dict["METADATA"]["ACTIVE"]
        metadata = query.get_file_entity(syn, metadata_synId)
        data = query.get_file_entity(syn, feature_synId)
        for test_type in data["test_type"].unique():
            subset = data[data["test_type"] == test_type]
            subset = groupby_wrapper(subset,
                                     "healthCode",
                                     metadata_cols)
            subset = pd.merge(subset, metadata, on="healthCode", how="inner")
            query.save_data_to_synapse(
                syn=syn,
                data=subset,
                source_table_id=[feature_synId, metadata_synId],
                used_script=used_script_url,
                output_filename=("grouped_%s_%s_features.csv" %
                                 (key, test_type)).lower(),
                data_parent_id="syn21537421")


if __name__ == '__main__':
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
