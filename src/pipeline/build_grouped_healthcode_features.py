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
    "FEATURE_DATA_SYNIDS": {
        "MPOWER_V1": "syn21597373",
        "MPOWER_V2": "syn21597625",
        "MPOWER_PASSIVE": "syn21597842",
        "ELEVATE_MS": "syn21597862"},
    "DEMOGRAPHIC_DATA_SYNID": "syn21602828",
    "OUTPUT_INFO": {
        "PARENT_FOLDER": "syn21592268",
        "PROJ_REPO_NAME": "mpower-gait-analysis",
        "PATH_GITHUB_TOKEN": "~/git_token.txt"}
}
syn = sc.login()

# TODO: fix using regex


def annot_phone(params):
    """
    Function to have more concrete phone types

    Args:
        params (type: string): raw phone information

    Returns:
        Rtype: String
        Returns an annotated dataset with lesser choice of phone information
    """
    if ";" in params:
        params = params.split(";")[0]
    params = params.replace(" ", "")
    if ("iPhone6+" in params):
        return "iPhone6+"
    elif ("iPhone6" in params):
        return "iPhone6"
    elif ("iPhone5" in params):
        return "iPhone5"
    elif ("iPhone8" in params):
        return "iPhone 8"
    elif ("iPhone9" in params):
        return "iPhone 9"
    elif ("iPhoneX" in params) or ("iPhone10" in params):
        return "iPhone X"
    else:
        return "Other iPhone"


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


def groupby_wrapper(data, group, metadata_columns=[]):
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

    # groupby features based on several aggregation
    feature_mapping = {"nonrot_seq": data[data["rotation_omega"].isnull()]
                       .drop("rotation_omega", axis=1),
                       "rot_seq": data[~data["rotation_omega"].isnull()]}
    for gait_sequence, feature_data in feature_mapping.items():
        feature_cols = [feat for feat in feature_data.columns if
                        (feat not in metadata_columns)
                        or (feat == "healthCode")]
        feature_mapping[gait_sequence] = feature_data[feature_cols]\
            .groupby(group)\
            .agg([np.max,
                  np.median,
                  np.mean,
                  q25, q75,
                  valrange, iqr])
        agg_feature_cols = []
        for feat, agg in feature_mapping[gait_sequence].columns:
            agg_feature_cols.append("{}_{}_{}"
                                    .format(gait_sequence, agg, feat))
        feature_mapping[gait_sequence].columns = agg_feature_cols

    feature_data = pd.concat([seqs for _, seqs in feature_mapping.items()],
                             join="outer",
                             axis=1)
    feature_data.index.name = "healthCode"

    # groupby metadata based on modes
    metadata = data[metadata_columns]\
        .groupby(["healthCode"])\
        .agg({"recordId": pd.Series.nunique,
              "phoneInfo": pd.Series.mode,
              "table_version": pd.Series.mode,
              "test_type": pd.Series.mode})
    metadata = metadata.rename({"recordId": "nrecords"}, axis=1)
    metadata["phoneInfo"] = metadata["phoneInfo"].apply(
        lambda x: x[0] if not isinstance(x, str) else x)
    metadata["phoneInfo"] = metadata["phoneInfo"].apply(annot_phone)

    # return features and metadata
    return feature_data.join(metadata, on="healthCode")


def main():
    used_script_url = query.get_git_used_script_url(
        path_to_github_token=data_dict["OUTPUT_INFO"]["PATH_GITHUB_TOKEN"],
        proj_repo_name=data_dict["OUTPUT_INFO"]["PROJ_REPO_NAME"],
        script_name=__file__)

    metadata_cols = ['appVersion', 'createdOn',
                     'phoneInfo', 'recordId',
                     'table_version', 'test_type',
                     'error_type', "healthCode"]
    demo_data = query.get_file_entity(
        syn, data_dict["DEMOGRAPHIC_DATA_SYNID"]).set_index("healthCode")
    for key, synId in data_dict["FEATURE_DATA_SYNIDS"].items():
        data = query.get_file_entity(syn, synId)
        for test_type in data["test_type"].unique():
            subset = data[data["test_type"] == test_type]
            subset = groupby_wrapper(subset,
                                     "healthCode",
                                     metadata_cols)
            subset = demo_data\
                .join(subset, on="healthCode", how="inner")\
                .reset_index()
            subset = subset.reset_index()
            query.save_data_to_synapse(
                syn=syn,
                data=subset,
                source_table_id=[synId, data_dict["DEMOGRAPHIC_DATA_SYNID"]],
                used_script=used_script_url,
                output_filename=("grouped_%s_%s_features.csv" %
                                 (key, test_type)).lower(),
                data_parent_id="syn21537421")


if __name__ == '__main__':
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
