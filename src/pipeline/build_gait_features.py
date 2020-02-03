## import future function ##
from __future__ import print_function
from __future__ import unicode_literals

## import standard library ##
import sys
import time
import argparse
import multiprocessing
import synapseclient as sc
import pandas as pd
import numpy as np
import synapseclient as sc

## import local modules ##
from utils import query_utils as query
from utils import gait_features_utils as gf_utils


## GLOBAL VARIABLES ## 
data_dict = {}
data_dict["GAIT_MPOWER_V1_TABLE"]      = {"synId": "syn10308918", 
                                          "table_version": "MPOWER_V1"} 
data_dict["GAIT_MPOWER_V2_TABLE"]      = {"synId": "syn12514611", 
                                          "table_version": "MPOWER_V2"}
data_dict["GAIT_MPOWER_PASSIVE_TABLE"] = {"synId": "syn17022539", 
                                          "table_version": "MPOWER_PASSIVE"}
data_dict["GAIT_EMS_TABLE"]            = {"synId": "syn10278766", 
                                          "table_version": "ELEVATE_MS"}

syn = sc.login()

def read_args():
    """
    Function for parsing in argument given by client
    returns argument parser object
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--update", action = "store_true",
                        help = "If specified, will update data based on new recordIds, otherwise start fresh")
    parser.add_argument("--cores", default= multiprocessing.cpu_count(),
                        help = "Number of Cores, negative number not allowed")
    parser.add_argument("--partition", default= 250,
                        help = "Number of sample per partition, no negative number")
    args = parser.parse_args()
    return args


def standardize_mpower_data(values):
    """
    Helper function for concatenating several synapse table columns into standardized name
    with annotation of which test it conducts, and the synapse table entity is it
    
    Cleaning Process: 
        -> Metadata feature columns will be collected in synapseTable (appVersion, phoneInfo, healthCode etc)
        -> For each filepath synapseTable column (deviceMotion, walking_motion, etc.) all names will be STANDARDIZED to
            gait_features.json_pathfile 
        -> Test type and version numbers will be annotated on each recordIds for persisting test information 

    Args:
        values (type: dict): A value from a dictionary containing synId and table version of 
                             queried mPower data from synapseTable
    
    Return:
        Rtype: pd.DataFrame
        Concattenated and cleaned dataset with filepaths to .synaseCache columns (gait_json_pathfile), 
        and metadata columns
    """
    
    table_id = values["synId"]
    table_version = values["table_version"]
    
    data = query.get_walking_synapse_table(syn = syn, 
                                            table_id      = table_id, 
                                            table_version = table_version, 
                                            retrieveAll   =  True)
    
    
    metadata = ["appVersion", "phoneInfo", "healthCode", "recordId", "createdOn"]
    filepath_cols = [cols for cols in data.columns if (cols not in metadata)]
    data_dict = {}
    for filepath in filepath_cols:
        data_dict[filepath] = data[(metadata + [filepath])]
        if (("rest" not in filepath) and ("balance" not in filepath)):
            print("walk: %s" %filepath)
            data_dict[filepath]["test_type"] = "walking"
        else:
            print("balance: %s" %filepath)
            data_dict[filepath]["test_type"] = "balance"
        data_dict[filepath].columns = ["gait_json_pathfile" if ((cols not in metadata) and (cols != "test_type"))\
                                       else cols for cols in data_dict[filepath].columns]
    concat_data = pd.concat([values for key, values in data_dict.items()]).reset_index(drop = True)
    concat_data["table_version"] = table_version
    return concat_data  

def clean_feature_sets(data, target_feature):
    metadata_feature = ['recordId', 'healthCode','appVersion', 
                        'phoneInfo', 'createdOn', 'test_type', 
                        "table_version"]
    feature_cols = metadata_feature + [target_feature]
    data = data[data[target_feature] != "#ERROR"][feature_cols]
    data = query.normalize_list_dicts_to_dataframe_rows(data, [target_feature])
    return data
    
def main():
    gaitfeatures = gf_utils.GaitFeaturize()
    syn = sc.login()
    args = read_args() 
    data = pd.concat([standardize_mpower_data(values) for key, 
                      values in data_dict.items()]).reset_index(drop = True)
    
    prev_stored_rotation_data = pd.DataFrame()
    prev_stored_walk_data     = pd.DataFrame()
    processed_records         = pd.DataFrame()
    
    if args.update:
        print("\n#########  UPDATING DATA  ################\n")
        processed_records         = query.check_children(syn = syn,
                                                        data_parent_id = "syn21537420", 
                                                        filename = "processed_records.csv")
        prev_stored_rotation_data = query.check_children(syn = syn, 
                                                        data_parent_id = "syn21537420", 
                                                        filename = "rotational_gait_features.csv")
        prev_stored_walk_data     = query.check_children(syn = syn, 
                                                        data_parent_id = "syn21537420",
                                                        filename = "nonrotational_gait_features.csv")
        data = data[~data["recordId"].isin(processed_records["recordId"].unique())]
        print("new rows that will be stored: {}".format(data.shape[0]))
    print("dataset combined, total rows for processing job are %s" %data.shape[0])

    ## featurize data ##
    data = query.parallel_func_apply(data, gaitfeatures.featurize_wrapper, 
                                    int(args.cores), int(args.partition)) 
    
    
    # cleaned_rotation_data = data[data["gait_rotation_features"] != "#ERROR"].drop(["gait_walk_features"], axis = 1)
    # cleaned_rotation_data = query.normalize_list_dicts_to_dataframe_rows(cleaned_rotation_data, ["gait_rotation_features"])
    # rotation_feature = [feat for feat in cleaned_rotation_data.columns if ("rotation" in feat) and ("pathfile" not in feat)]
    # features = metadata_feature + rotation_feature
    cleaned_rotation_data = clean_feature_sets(data, "gait_rotation_features")
    cleaned_rotation_data = pd.concat([prev_stored_rotation_data, cleaned_rotation_data]).reset_index(drop = True)

    query.save_data_to_synapse(syn = syn, 
                            data = cleaned_rotation_data, 
                            source_table_id =  [values["synId"] for key, values in data_dict.items()],
                            output_filename = "rotation_gait_features.csv",
                            data_parent_id = "syn21537420")
    
    print("\n################################## Saved Rotation Data ######################################\n")
    
    # cleaned_walk_data = data[data["gait_walk_features"] != "#ERROR"].drop(["gait_rotation_features"], axis = 1)
    # cleaned_walk_data = query.normalize_list_dicts_to_dataframe_rows(cleaned_walk_data, ["gait_walk_features"])
    # walking_feature = [feat for feat in cleaned_walk_data.columns if ("walking" in feat) and ("pathfile" not in feat)]
    # features = metadata_feature + walking_feature
    cleaned_walk_data = clean_feature_sets(data, "gait_walk_features")
    cleaned_walk_data = pd.concat([prev_stored_walk_data, cleaned_walk_data]).reset_index(drop = True)

    query.save_data_to_synapse(syn = syn, 
                                data = cleaned_walk_data, 
                                source_table_id = [values["synId"] for key, values in data_dict.items()],
                                output_filename = "walking_gait_features.csv",
                                data_parent_id  = "syn21537420") 
    
    print("\n################################## Saved Walking Data ######################################\n") 
    

    new_records = data[["recordId"]].drop_duplicates(keep = "first").reset_index(drop = True)
    processed_records = pd.concat([processed_records, new_records]).reset_index(drop = True)

    query.save_data_to_synapse(syn = syn,
                                data = processed_records,
                                source_table_id = [values["synId"] for key, values in data_dict.items()],
                                output_filename = "processed_records.csv",
                                data_parent_id  = "syn21537420")
    
    print("\n################################## Saved Processed RecordIds Logging ########################\n") 
    

if __name__ ==  '__main__': 
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
