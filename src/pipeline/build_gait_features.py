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


def clean_gait_mpower_dataset(data, filepath_colname, test_type, table_version):
    """
    Helper function for cleaning queried dataset from SynapseTable;
    Cleaning Process: 
        -> Metadata feature columns will be collected in synapseTable (appVersion, phoneInfo, healthCode etc)
        -> For each filepath synapseTable column (deviceMotion, walking_motion, etc.) all names will be STANDARDIZED to
            gait_features.json_pathfile 
        -> Test type and version numbers will be annotated on each recordIds for persisting test information 

    Args:
        data    (type: pd.DataFrame): A dataframe consisting of metadata from synapseTable 
                                        and filepath to .synapseCace
        filepath_colname (type: str): Change column name to something that is consistent for analysis
        test_type        (type: str): Define whether test_type is walking or balance test
        table_version    (type: str): Define where data is sourced from ("elevate_ms", "mpower_v1", 
                                        "mpower_v2", "mpower_passive")
    
    Return:
        Rtype: pd.DataFrame
        Cleaned dataset with filepaths to .synaseCache columns, and metadata columns
    """
    metadata = ["appVersion", "phoneInfo", "healthCode", "recordId", "createdOn"]
    data = data[[feature for feature in data.columns if \
                            (filepath_colname in feature) or \
                            feature in metadata]]\
                            .rename({filepath_colname: "gait.json_pathfile"}, 
                                       axis = 1)
    data["test_type"] = test_type
    data["table_version"] = table_version
    return data

def main():
    gaitfeatures = gf_utils.GaitFeaturize()
    syn = sc.login()

    args = read_args() 
    metadata_feature = ['recordId', 'healthCode','appVersion', 
                        'phoneInfo', 'createdOn', 'test_type', "table_version"]

    ## retrieve from synapseTable (gait) version 2 ## 
    query_data_v1 = query.get_walking_synapse_table(syn               = syn, 
                                                    table_id          =  "syn10308918", 
                                                    table_version     =  "MPOWER_V1", 
                                                    retrieveAll       =  True)
    
    ## retrieve from synapseTable (gait) version 2 ## 
    query_data_v2 = query.get_walking_synapse_table(syn               = syn, 
                                                    table_id          = "syn12514611", 
                                                    table_version     = "MPOWER_V2", 
                                                    retrieveAll       = True)
    
    ## retrieve data from passive gait synapseTable ##
    query_data_passive = query.get_walking_synapse_table(syn          = syn,
                                                        table_id      = "syn17022539", 
                                                        table_version = "MPOWER_PASSIVE",
                                                        retrieveAll   = True)
    ## retrieve data from EMS gait synapse table
    query_data_ems = query.get_walking_synapse_table(syn              = syn,
                                                    table_id          = "syn10278766", 
                                                    table_version     = "ELEVATE_MS",
                                                    retrieveAll       = True)
    
    mpowerv1_data_outbound = clean_gait_mpower_dataset(data           = query_data_v1, 
                                                    filepath_colname  = "deviceMotion_walking_outbound.json.items_pathfile",
                                                    test_type         = "walking", 
                                                    table_version     = "MPOWER_V1")
    
    mpowerv1_data_return = clean_gait_mpower_dataset(data        = query_data_v1, 
                                                filepath_colname = "deviceMotion_walking_return.json.items_pathfile",
                                                test_type        = "walking", 
                                                table_version    = "MPOWER_V1")

    mpowerv1_data_balance1 = clean_gait_mpower_dataset(data       = query_data_v1, 
                                                filepath_colname = "deviceMotion_walking_rest.json.items_pathfile",
                                                test_type        = "balance", 
                                                table_version    = "MPOWER_V1")

    mpowerv1_data_balance2 = clean_gait_mpower_dataset(data       = query_data_v1, 
                                                filepath_colname = "deviceMotion_walking.rest.json.items_pathfile",
                                                test_type        = "balance", 
                                                table_version    = "MPOWER_V1")

    ems_data_outbound = clean_gait_mpower_dataset(data           = query_data_ems, 
                                                filepath_colname = "deviceMotion_walking_outbound.json.items_pathfile",
                                                test_type        = "walking", 
                                                table_version    = "ELEVATE_MS")
    
    ems_data_return = clean_gait_mpower_dataset(data             = query_data_ems, 
                                                filepath_colname = "deviceMotion_walking_return.json.items_pathfile",
                                                test_type        = "walking", 
                                                table_version    = "ELEVATE_MS")

    ems_data_balance = clean_gait_mpower_dataset(data            = query_data_ems, 
                                                filepath_colname = "deviceMotion_walking_rest.json.items_pathfile",
                                                test_type        = "balance", 
                                                table_version    = "ELEVATE_MS")

    mpowerv2_data_walking = clean_gait_mpower_dataset(data       = query_data_v2, 
                                                filepath_colname = "walk_motion.json_pathfile",
                                                test_type        = "walking", 
                                                table_version    = "MPOWER_V2")

    mpowerv2_data_balance = clean_gait_mpower_dataset(data       = query_data_v2, 
                                                filepath_colname = "balance_motion.json_pathfile",
                                                test_type        = "balance", 
                                                table_version    = "MPOWER_V2")

    mpowerpassive_data_walking = clean_gait_mpower_dataset(data  = query_data_passive, 
                                                filepath_colname = "walk_motion.json_pathfile",
                                                test_type        = "walking", 
                                                table_version    = "MPOWER_PASSIVE")

    ## concat all data into one collective dataframe ##
    data = pd.concat([mpowerv1_data_outbound, 
                  mpowerv1_data_return, 
                  mpowerv1_data_balance1,
                  mpowerv1_data_balance2, 
                  mpowerv2_data_walking, 
                  mpowerv2_data_balance,
                  mpowerpassive_data_walking,
                  ems_data_outbound, 
                  ems_data_return, 
                  ems_data_balance]).reset_index(drop = True).tail(55)
    
    prev_stored_rotation_data = pd.DataFrame()
    prev_stored_walk_data = pd.DataFrame()
    
    if args.update:
        print("\n#########  UPDATING DATA  ################\n")
        processed_records = query.check_children(syn = syn,
                                                data_parent_id = "syn21537420", 
                                                filename = "processed_records.csv")
        prev_stored_rotation_data = query.check_children(syn = syn, 
                                                        data_parent_id = "syn21537420", 
                                                        filename = "rotational_gait_features.csv")
        prev_stored_walk_data = query.check_children(syn = syn, 
                                                            data_parent_id = "syn21537420",
                                                            filename = "nonrotational_gait_features.csv")
        data = data[~data["recordId"].isin(processed_records["recordId"].unique())]
        print("new rows that will be stored: {}".format(data.shape[0]))
    print("dataset combined, total rows for processing job are %s" %data.shape[0])

    ## featurize data ##
    data = query.parallel_func_apply(data, gaitfeatures.featurize_wrapper, 
                                    int(args.cores), int(args.partition)) 
    
    ## concat previously stored data with new rows ## 
    #data = pd.concat([prev_stored_data, data]).reset_index(drop = True)

    #query.save_data_to_synapse(syn = syn, 
     #                           data = data, 
     #                           output_filename = "raw_gait_features.csv",
     #                           source_table_id =  [GAIT_EMS_TABLE, GAIT_MPOWER_V1_TABLE, 
     #                                               GAIT_MPOWER_V2_TABLE, GAIT_MPOWER_PASSIVE_TABLE],
     #                           data_parent_id  = "syn21537420")
    
    cleaned_rotation_data = data[data["gait_rotation_features"] != "#ERROR"].drop(["gait_walk_features"], axis = 1)
    cleaned_rotation_data = query.normalize_list_dicts_to_dataframe_rows(cleaned_rotation_data, ["gait_rotation_features"])
    rotation_feature = [feat for feat in cleaned_rotation_data.columns if ("rotation" in feat) and ("pathfile" not in feat)]
    features = metadata_feature + rotation_feature

    cleaned_rotation_data = pd.concat([prev_stored_rotation_data, cleaned_rotation_data]).reset_index(drop = True)

    query.save_data_to_synapse(syn = syn, 
                            data = cleaned_rotation_data[features], 
                            source_table_id =  [GAIT_EMS_TABLE, GAIT_MPOWER_V1_TABLE, 
                                                GAIT_MPOWER_V2_TABLE, GAIT_MPOWER_PASSIVE_TABLE],
                            output_filename = "rotational_gait_features.csv",
                            data_parent_id = "syn21537420")
    print("Saved rotation data") 
    
    cleaned_walk_data = data[data["gait_walk_features"] != "#ERROR"].drop(["gait_rotation_features"], axis = 1)
    cleaned_walk_data = query.normalize_list_dicts_to_dataframe_rows(cleaned_walk_data, ["gait_walk_features"])
    walking_feature = [feat for feat in cleaned_walk_data.columns if ("walking" in feat) and ("pathfile" not in feat)]
    features = metadata_feature + walking_feature
    
    cleaned_walk_data = pd.concat([prev_stored_walk_data, cleaned_walk_data]).reset_index(drop = True)

    query.save_data_to_synapse(syn = syn, 
                                data = cleaned_walk_data[features], 
                                source_table_id = [GAIT_EMS_TABLE, GAIT_MPOWER_V1_TABLE, 
                                                    GAIT_MPOWER_V2_TABLE, GAIT_MPOWER_PASSIVE_TABLE],
                                output_filename = "nonrotational_gait_features.csv",
                                data_parent_id  = "syn21537420") 
    print("Saved walking data") 

    query.save_data_to_synapse(syn = syn,
                                data = cleaned_walk_data[["recordId"]].drop_duplicates(keep = "first").reset_index(drop = True),
                                output_filename = "processed_records.csv",
                                data_parent_id  = "syn21537420")
    print("Saved processed records logging")
    

if __name__ ==  '__main__': 
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
