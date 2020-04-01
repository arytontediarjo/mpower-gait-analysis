"""
Script to group features by healthcodes or recordIds
with several aggregations methods, 

QC Steps:
-> Removing data of resting inferred from their freeze indexes
-> Records of less than 5 of each users are not considered
-> Aggregation will be based on their healthCode and test type,
so that records from each test e.g passive data and walking data
will not be mixed up
"""

# import future libraries
from __future__ import print_function
from __future__ import unicode_literals

# import standard libraries
import time
import os
import argparse
import pandas as pd
import numpy as np
import gc

# import external libraries
import synapseclient as sc

# import project modules
from utils import query_utils as query


# global variables
DATA_DICT = {
    "FEATURE_DATA_SYNIDS": {
        "MPOWER_V1": "syn21765655",
        "MPOWER_V2": "syn21765659",
        "MPOWER_PASSIVE": "syn21765662",
        "ELEVATE_MS": "syn21765671"},
    "DEMOGRAPHIC_DATA_SYNID": "syn21602828",
    "OUTPUT_INFO": {
        "PARENT_SYN_ID": "syn21537421",
        "PROJ_REPO": "mpower-gait-analysis",
        "TOKEN_PATH": "~/git_token.txt"}
}


def read_args():
    """
    Function for parsing in argument given by client
    returns argument parser object
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", default="healthCode",
                        choices=["healthCode", "recordId"],
                        help="type of aggregation")
    args = parser.parse_args()
    return args


# TODO: fix using regex
def annot_phone(params):
    """
    Function to simplify phone types
    Args:
        params (type: string): raw phone information
    Returns:
        Rtype: String
        Returns an annotated dataset with lesser choice of phone information
    """
    if ";" in params:
        params = params.split(";")[0]
    params = params.replace(" ", "")
    if ("iPhone6+" in params) or ("iPhone6S+" in params):
        return "iPhone6+"
    elif ("iPhone6" in params):
        return "iPhone6"
    elif ("iPhone5" in params):
        return "iPhone5"
    elif ("iPhone8" in params):
        return "iPhone8"
    elif ("iPhone9" in params):
        return "iPhone9"
    elif ("iPhoneX" in params) or ("iPhone10" in params):
        return "iPhoneX"
    else:
        return "Others"


def iqr(x):
    """
    Function for getting IQR value
    """
    return x.quantile(0.75) - x.quantile(0.25)


def aggregate_wrapper(data, group, metadata_columns=[]):
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

    data = data[data["gait_segment"] != "rest"]

    # groupby features based on several aggregation
    feature_mapping = {
        "nonrot_data":
        data[(data["gait_segment"] == "walk")
             & (data["window_size"] >= 5)]
        .drop("rotation_omega", axis=1),
        "rot_data":
        data[data["gait_segment"] == "rotation"][[group, "rotation_omega"]]
    }
    for gait_sequence, feature_data in feature_mapping.items():
        feature_cols = [feat for feat in feature_data.columns if
                        (feat not in metadata_columns)
                        or (feat == group)]
        feature_mapping[gait_sequence] = feature_data[feature_cols]\
            .groupby(group)\
            .agg([np.max,
                  np.median,
                  np.std,
                  iqr])
        agg_feature_cols = []
        for feat, agg in feature_mapping[gait_sequence].columns:
            agg_feature_cols.append("{}_{}"
                                    .format(agg, feat))
        feature_mapping[gait_sequence].columns = agg_feature_cols

    feature_data = pd.concat([seqs for _,
                              seqs in feature_mapping.items()],
                             join="outer",
                             axis=1)
    feature_data.index.name = group
    feature_data = feature_data.reset_index()

    # if aggregate on recordId no need to aggregate metadata
    if group == "recordId":
        metadata = data[metadata_columns].\
            drop_duplicates(subset=["recordId"],
                            keep="first")

    # aggregate on healthcode require aggregate on metadata
    else:
        metadata = data[metadata_columns]\
            .groupby([group])\
            .agg({"recordId": pd.Series.nunique,
                  "phoneInfo": pd.Series.mode,
                  "table_version": pd.Series.mode,
                  "test_type": pd.Series.mode})
        metadata = metadata.rename(
            {"recordId": "nrecords"}, axis=1)
        metadata = metadata.reset_index()
    metadata["phoneInfo"] = metadata["phoneInfo"].apply(
        lambda x: x[0] if not isinstance(x, str) else x)
    metadata["phoneInfo"] = metadata["phoneInfo"].apply(annot_phone)

    feature_data = pd.merge(
        feature_data, metadata, on=group, how="left")
    return feature_data


def main():
    """
    Main Function:

    Takes in several data from featurized file entity and demographics
    file entity and group by based on user prompt (recordId or healthcodes)
    using aggregation based on interquartiles,
    value ranges, median and abs max.
    """

    args = read_args()
    # retrieve synapse credential through config
    path = os.path.join(os.getenv("HOME"),
                        ".synapseConfig")
    syn = sc.Synapse(configPath=path)
    syn.login(os.getenv("syn_username"),
              os.getenv("syn_password"),
              rememberMe=True)

    metadata_cols = ['appVersion', 'createdOn',
                     'phoneInfo', 'recordId',
                     'table_version', 'test_type',
                     'error_type', "healthCode"]
    demo_data = query.get_file_entity(
        syn, DATA_DICT["DEMOGRAPHIC_DATA_SYNID"])
    results_group_data = pd.DataFrame()
    for _, synId in DATA_DICT["FEATURE_DATA_SYNIDS"].items():
        data = query.get_file_entity(syn, synId)
        for test_type in data["test_type"].unique():
            subset = data[data["test_type"] == test_type]
            grouped_feature_data = aggregate_wrapper(
                subset,
                args.group,
                metadata_cols)
            grouped_feature_data = pd.merge(
                grouped_feature_data, demo_data,
                on="healthCode", how="left")
            results_group_data = pd.concat(
                [results_group_data, grouped_feature_data])\
                .reset_index(drop=True)

        # delete unused data to save memory usage
        del data
        del grouped_feature_data
        del subset
        gc.collect()

    used_script_url = query.get_git_used_script_url(
        path_to_github_token=DATA_DICT["OUTPUT_INFO"]["TOKEN_PATH"],
        proj_repo_name=DATA_DICT["OUTPUT_INFO"]["PROJ_REPO"],
        script_name=__file__)

    query.save_data_to_synapse(
        syn=syn,
        data=results_group_data,
        source_table_id=[synid for key, synid in
                         DATA_DICT["FEATURE_DATA_SYNIDS"].items()]
        + [DATA_DICT["DEMOGRAPHIC_DATA_SYNID"]],
        used_script=used_script_url,
        output_filename=("filtered_grouped_%s_features.csv" %
                         (args.group)),
        data_parent_id=DATA_DICT["OUTPUT_INFO"]["PARENT_SYN_ID"])


if __name__ == '__main__':
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
