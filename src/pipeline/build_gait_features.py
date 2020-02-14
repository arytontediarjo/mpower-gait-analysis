"""
Author: Sage Bionetworks

About:
A data pipeline script for extracting all the gait data from
Sage Bionetworks Synapse Table (MPowerV1, MPowerV2,
MPower Passive, Elevate MS), featurize data based on rotational
features and features from PDKit (external source).
Result of this data pipeline will all be saved as Synapse File Entity.
"""

# import future function
from __future__ import print_function
from __future__ import unicode_literals

# import standard library
import time
import gc
import argparse
import multiprocessing
import synapseclient as sc
import pandas as pd

# import local modules
from utils import query_utils as query
from utils import gait_features_utils as gf_utils


# GLOBAL VARIABLES
data_dict = {"MPOWER_V1": {"synId": "syn10308918",
                           "table_version": "MPOWER_V1"},
             "MPOWER_V2": {"synId": "syn12514611",
                           "table_version": "MPOWER_V2"},
             "MPOWER_PASSIVE": {"synId": "syn17022539",
                                "table_version": "MPOWER_PASSIVE"},
             "ELEVATE_MS": {"synId": "syn10278766",
                            "table_version": "ELEVATE_MS"},
             "OUTPUT_INFO": {"parent_folder_synId": "syn21537420",
                             "proj_repo_name": "mpower-gait-analysis",
                             "git_token_path": "~/git_token.txt"}
             }
syn = sc.login()


def read_args():
    """
    Function for parsing in argument given by client
    returns argument parser object
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--update", action="store_true",
                        help="If specified, will update data based on new recordIds,\
                                otherwise start fresh")
    parser.add_argument("--cores", default=multiprocessing.cpu_count(),
                        help="Number of Cores, negative number not allowed")
    parser.add_argument("--partition", default=250,
                        help="Number of sample per partition,\
                            no negative number")
    args = parser.parse_args()
    return args


def featurize_wrapper(data):
    """
    wrapper function for multiprocessing jobs
    Args:
        data (type: pd.DataFrame): takes in pd.DataFrame
    returns a json file featurized walking data
    """
    gaitfeatures = gf_utils.GaitFeaturize()
    data["gait_features"] = data["gait_json_filepath"].apply(
        gaitfeatures.run_gait_feature_pipeline)
    return data


def standardize_mpower_data(values):
    """
    Helper function for concatenating several synapse table columns into
    standardized name with annotation of which test it conducts,
    and the version of synapse table entity it is

    Cleaning Process:
        -> Metadata feature columns will be collected
        in synapseTable (appVersion, phoneInfo, healthCode etc)
        -> For each filepath synapseTable column (deviceMotion,
        walking_motion,etc.) all names will be standardized
        to gait_json_filepath
        -> Test type and version numbers will be annotated
        on each recordIds for persisting test information

    Args:
        values (type: dict): A value from a dictionary containing
        synId and table version of queried mPower data from synapseTable

    Return:
        Rtype: pd.DataFrame
        Concattenated and cleaned dataset with filepaths to json files as
        columns (gait_json_pathfile), and metadata columns
    """
    table_id = values["synId"]
    table_version = values["table_version"]

    data = query.get_walking_synapse_table(syn=syn,
                                           table_id=table_id,
                                           table_version=table_version,
                                           retrieveAll=True)

    metadata = ["appVersion", "phoneInfo",
                "healthCode", "recordId", "createdOn"]
    filepath_cols = [cols for cols in data.columns if (cols not in metadata)]
    data_dict = {}
    for filepath in filepath_cols:
        data_dict[filepath] = data[(metadata + [filepath])]
        if (("rest" not in filepath) and ("balance" not in filepath)):
            data_dict[filepath]["test_type"] = "walking"
        else:
            data_dict[filepath]["test_type"] = "balance"
        data_dict[filepath].columns = ["gait_json_filepath" if
                                       ((cols not in metadata)
                                        and (cols != "test_type"))
                                       else cols for cols
                                       in data_dict[filepath].columns]
    concat_data = pd.concat(
        [values for key, values in data_dict.items()]).reset_index(drop=True)
    concat_data["table_version"] = table_version
    return concat_data


def normalize_feature_sets(data, target_feature):
    """
    Utility function normalize feature into several rows,
    clean feature sets, and concattenate error messages
    Args:
      data (pd.DataFrame)    : dataframe of concattenated data
      target_feature (string): dataframe features
    Return:
      RType: pd.DataFrame
      Returns a dataframe with normalized columns and metadata
    """
    metadata_feature = ['recordId', 'healthCode', 'appVersion',
                        'phoneInfo', 'createdOn', 'test_type',
                        "table_version"]

    # get feature data
    feature_cols = metadata_feature + [target_feature]
    nonerror_data = data[data[target_feature]
                         .apply(lambda x: isinstance(x, list))][feature_cols]
    feature_data = query.norm_list_dicts_to_rows(
        nonerror_data, [target_feature])

    # get error messages
    error_data = data[data[target_feature].apply(
        lambda x: isinstance(x, str))][feature_cols]
    error_data = error_data.rename({"gait_features": "error_type"}, axis=1)

    # combine data
    data = pd.concat([error_data, feature_data],
                     sort=False).reset_index(drop=True)
    return data


def main():
    args = read_args()
    all_data = pd.concat([standardize_mpower_data(values) for key,
                          values in data_dict.items()
                          if key != "OUTPUT_INFO"]).reset_index(drop=True)

    # Iteratively save data in different tables, join after in synapse
    # Garbage collect the data so that it does not overload
    for version in all_data["table_version"].unique():
        data = all_data[all_data["table_version"] == version]
        prev_stored_data = pd.DataFrame()
        if args.update:
            prev_stored_data = query.check_children(
                syn=syn,
                data_parent_id=data_dict["OUTPUT_INFO"]["parent_folder_synId"],
                filename="%s_gait_features.csv" % version)
            data = data[~data["recordId"].isin(
                prev_stored_data["recordId"].unique())]
        # featurize data if not empty
        if not data.empty:
            data = query.parallel_func_apply(
                data, featurize_wrapper, int(args.cores), int(args.partition))
            data = normalize_feature_sets(data, "gait_features")

        # concat data to previously stored data
        data = pd.concat([prev_stored_data, data])\
            .reset_index(drop=True)
        query.save_data_to_synapse(
            syn=syn,
            data=data,
            used_script=query.get_git_used_script_url(
                path_to_github_token=data_dict["OUTPUT_INFO"]["git_token_path"],
                proj_repo_name=data_dict["OUTPUT_INFO"]["proj_repo_name"],
                script_name=__file__),
            source_table_id=data_dict[version]["synId"],
            output_filename="%s_gait_features.csv" % version,
            data_parent_id=data_dict["OUTPUT_INFO"]["parent_folder_synId"])

        # clear data from memory
        del data
        gc.collect()


if __name__ == '__main__':
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
