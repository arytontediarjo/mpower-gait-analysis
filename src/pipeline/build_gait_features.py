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
GAIT_MPOWER_V1_TABLE      =  "syn10308918"
GAIT_MPOWER_V2_TABLE      =  "syn12514611"
GAIT_MPOWER_PASSIVE_TABLE =  "syn17022539"
GAIT_EMS_TABLE            =  "syn10278766"


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


def concat_mpower_datasets(values):
    """
    Helper function for concatenating data from mPower synapseTable
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

def main():
    gaitfeatures = gf_utils.GaitFeaturize()
    syn = sc.login()
    metadata_feature = ['recordId', 'healthCode','appVersion', 
                        'phoneInfo', 'createdOn', 'test_type', 
                        "table_version"]

    args = read_args() 
    data = pd.concat([concat_mpower_datasets(values) for key, 
                      values in data_dict.items()]).reset_index(drop = True)
    
    prev_stored_rotation_data = pd.DataFrame()
    prev_stored_walk_data     = pd.DataFrame()
    
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
    
    cleaned_rotation_data = data[data["gait_rotation_features"] != "#ERROR"].drop(["gait_walk_features"], axis = 1)
    cleaned_rotation_data = query.normalize_list_dicts_to_dataframe_rows(cleaned_rotation_data, ["gait_rotation_features"])
    rotation_feature = [feat for feat in cleaned_rotation_data.columns if ("rotation" in feat) and ("pathfile" not in feat)]
    features = metadata_feature + rotation_feature

    cleaned_rotation_data = pd.concat([prev_stored_rotation_data, cleaned_rotation_data]).reset_index(drop = True)

    query.save_data_to_synapse(syn = syn, 
                            data = cleaned_rotation_data[features], 
                            source_table_id =  [GAIT_EMS_TABLE, GAIT_MPOWER_V1_TABLE, 
                                                GAIT_MPOWER_V2_TABLE, GAIT_MPOWER_PASSIVE_TABLE],
                            output_filename = "rotation_gait_features.csv",
                            data_parent_id = "syn21537420")
    print("################################## Saved Rotation Data ######################################")
    
    cleaned_walk_data = data[data["gait_walk_features"] != "#ERROR"].drop(["gait_rotation_features"], axis = 1)
    cleaned_walk_data = query.normalize_list_dicts_to_dataframe_rows(cleaned_walk_data, ["gait_walk_features"])
    walking_feature = [feat for feat in cleaned_walk_data.columns if ("walking" in feat) and ("pathfile" not in feat)]
    features = metadata_feature + walking_feature
    
    cleaned_walk_data = pd.concat([prev_stored_walk_data, cleaned_walk_data]).reset_index(drop = True)

    query.save_data_to_synapse(syn = syn, 
                                data = cleaned_walk_data[features], 
                                source_table_id = [GAIT_EMS_TABLE, GAIT_MPOWER_V1_TABLE, 
                                                    GAIT_MPOWER_V2_TABLE, GAIT_MPOWER_PASSIVE_TABLE],
                                output_filename = "walking_gait_features.csv",
                                data_parent_id  = "syn21537420") 
    print("################################## Saved Walking Data ######################################") 

    query.save_data_to_synapse(syn = syn,
                                data = cleaned_walk_data[["recordId"]].drop_duplicates(keep = "first").reset_index(drop = True),
                                source_table_id = [GAIT_EMS_TABLE, GAIT_MPOWER_V1_TABLE, 
                                                    GAIT_MPOWER_V2_TABLE, GAIT_MPOWER_PASSIVE_TABLE],
                                output_filename = "processed_records.csv",
                                data_parent_id  = "syn21537420")
    print("################################## Saved Processed RecordIds Logging ###################################") 
    

if __name__ ==  '__main__': 
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
