"""
Author: Sage Bionetworks

Utility functions used for Sage Bionetworks data pipeline process,
starting from table querying, setting provenance, updating data, 
and updating git repo (TODO)
"""


## future library imports ## 
from __future__ import unicode_literals
from __future__ import print_function

## standard library imports ##
import sys
import subprocess
from github import Github
import json 
import os
import pandas as pd
import numpy as np
import multiprocessing as mp

## external library imports ##
import synapseclient as sc
from synapseclient import (Entity, Project, Folder, File, Link, Activity)


def get_walking_synapse_table(syn, 
                            table_id, 
                            table_version, 
                            healthCodes = None, 
                            recordIds = None, 
                            retrieveAll = False):
    """
    Utility function to query synapse walking table entity, 
    can be used for all table versions given a parameter of 
    lists of user healthcodes and recordIds; 
    retrieve all parameter can be set to true if required to query 
    all the gait database
    -> Querying process:
        1.) Query data usings synapse tableQuery on non-iOS data (for now), then parse it to dataframe
        2.) mpower v1 and elevate MS "device_motion" data will be taken, and all json files from mpowerV2 and mpower_passive will be taken
        3.) Use synapse downloadTableColumns to download all the designated column files from the table
        4.) Inner join the parsed dataframe from synapse table query dataframe with download Table Columns by their file handle ids
    
    Args:  
        syn              : synapse object,             
        table_id    (str): id of table entity,
        version     (str): version number (args (string) = ["MPOWER_V1", "MPOWER_V2", "MS_ACTIVE", "PASSIVE"])
        healthcodes (str): list or array of healthcodes
        recordIDs   (str): list or of recordIds
    
    Return: 
        RType: pd.DataFrame
        returns a pandas dataframe of recordIds and their respective metadata, 
        alongside their file handle ids and file paths with empty file paths annotated to "#ERROR"
    """
    print("Querying %s Data" %table_version)

    if not retrieveAll:
        if not isinstance(recordIds, type(None)):
            recordId_subset = "({})".format([i for i in recordIds]).replace("[", "").replace("]", "")
            query = syn.tableQuery("select * from {} WHERE recordId in {}".format(table_id, recordId_subset))
        else:
            healthCode_subset = "({})".format([i for i in healthCodes]).replace("[", "").replace("]", "")
            query = syn.tableQuery("select * from {} WHERE healthCode in {}".format(table_id, healthCode_subset))
    else:
        query = syn.tableQuery("select * from {} where \
                                phoneInfo like '%iPhone%'\
                                or phoneInfo like '%iOS%'".format(table_id))
    data = query.asDataFrame()
    
    ## unique table identifier in mpowerV1 and EMS synapse table
    if (table_version == "MPOWER_V1") or (table_version == "ELEVATE_MS"):
        column_list = [_ for _ in data.columns if ("deviceMotion" in _)]
    ## unique table identifier in mpowerV2 and passive data
    elif (table_version == "MPOWER_V2") or (table_version == "MPOWER_PASSIVE") :
        column_list = [_ for _ in data.columns if ("json" in _)]
    ## raise error if version is not recognized
    else:
        raise Exception("version type is not recgonized, \
                        please use either of these choices:\
                        (MPOWER_V1, MS_ACTIVE, MPOWER_V2, MPOWER_PASSIVE)")
    
    ## download columns that contains walking data based on the logical condition
    file_map = syn.downloadTableColumns(query, column_list)
    dict_ = {}
    dict_["file_handle_id"] = []
    dict_["file_path"] = []
    for k, v in file_map.items():
        dict_["file_handle_id"].append(k)
        dict_["file_path"].append(v)
    filepath_data = pd.DataFrame(dict_)
    data = data[["recordId", "healthCode", 
                "appVersion", "phoneInfo", 
                "createdOn"] + column_list]
    filepath_data["file_handle_id"] = filepath_data["file_handle_id"].astype(float)
    
    ### Join the filehandles with each acceleration files ###
    for feat in column_list:
        data[feat] = data[feat].astype(float)
        data = pd.merge(data, filepath_data, 
                        left_on = feat, 
                        right_on = "file_handle_id", 
                        how = "left")
        data = data.rename(columns = {feat: "{}_path_id".format(feat), 
                                    "file_path": "{}_pathfile".format(feat)})\
                                        .drop(["file_handle_id"], axis = 1)
    ## Empty Filepaths on synapseTable ##
    cols = [feat for feat in data.columns if "path_id" not in feat]
    return data[cols]



def save_data_to_synapse(syn,
                        data, 
                        output_filename,
                        data_parent_id, 
                        used_script     = None,
                        source_table_id = None,
                        remove          = True): 
    """
    Utility function to save data to synapse given a parent id, list of used script, 
    and list of source table where the query was sourced
    
    Args: 
        syn                              = synapse object        
        data (pd.DataFrame)              = tabular data, script or notebook 
        output_filename (string)         = the name of the output file 
        data_parent_id (list, np.array)  = the parent synid where data will be stored 
        used_script (list, np.array)     = git repo url that produces this data (if available)
        source_table_id (list, np.array) = list of source of where this data is produced (if available) 
        remove (boolean)                 = prompt to remove data after saving
    
    Returns:
        Returns stored file entity in Synapse database
    """
    ## path to output filename for reference ##
    path_to_output_filename = os.path.join(os.getcwd(), output_filename)
        
    ## save the script to synapse ##
    if isinstance(data, pd.DataFrame):
        data = data.to_csv(path_to_output_filename)
    
    ## create new file instance and set up the provenance
    new_file = File(path = path_to_output_filename, parentId = data_parent_id)
        
    ## instantiate activity object
    act = Activity()
    if source_table_id is not None:
        act.used(source_table_id)
    if used_script is not None:
        act.executed(used_script)
        
    ## store to synapse ## 
    new_file = syn.store(new_file, activity = act)           
        
    ## remove the file ##
    if remove:
        os.remove(path_to_output_filename)

  
def normalize_dict_to_column_features(data, features):
    """
    Utiltiy function to normalize column that contains dictionaries into separate columns
    in the dataframe, for any null values, it will be annotated as "#ERROR"
    
    Args: 
        data (type: pd.DataFrame)  : pandas DataFrame       
        features (string)          : list of dict target features for normalization 
    
    Returns:
        Rtype: pd.DataFrame
        Returns a normalized dataframe with column containing normalized dictionary that will span to 
        dataframe columns
    """
    for feature in features:
        normalized_data = data[feature].map(lambda x: x if isinstance(x, dict) else "#ERROR") \
                                    .apply(pd.Series) \
                                    .fillna("#ERROR") \
                                    .add_prefix('{}.'.format(feature))
        data = pd.concat([data, normalized_data], axis = 1).drop([feature, "%s.0"%feature], axis = 1)
    return data

def normalize_list_dicts_to_dataframe_rows(data, features):
    """
    Utility function to normalize a column that contains list of dictionaries into 
    separate rows
    
    Args:
        data (pd.DataFrame): pandas DataFrame
        features   (string): a list of features for normalization to rows
    
    Returns: 
        A normalized dataframe with rows from normalized from list of dictionaries
    """
    for feature in features:
        data = (pd.concat({i: pd.DataFrame(x) for i, x in data[feature].items()})
                 .reset_index(level=1, drop=True)
                 .join(data).drop([feature], axis = 1)
                 .reset_index(drop = True)
                )
    return data

 
def get_file_entity(syn, synid):
    """
    Utility function to get data (csv,tsv) file entity and turn it 
    into pd.DataFrame
    Args:
        syn   : a syn object
        synid : syn id of file entity
    """
    entity = syn.get(synid)
    if (".tsv" in entity["name"]):
        separator = "\t"
    else:
        separator = ","
    data = pd.read_csv(entity["path"],index_col = 0, sep = separator)
    return data


def parallel_func_apply(df, func, no_of_processors, chunksize):
    """
    Utility function for parallelizing pd.DataFrame processing
    parameter: 
         df               = pandas DataFrame         
         func             = wrapper function for data processing
         no_of_processors = number of processors to transform the data
         chunksize        = number of partition 
    Returns: 
        RType: pd.DataFrame
        Returns a transformed dataframe from the wrapper apply function 
    """
    df_split = np.array_split(df, chunksize)
    print("Currently running on {} processors".format(no_of_processors))
    pool = mp.Pool(no_of_processors)
    map_values = pool.map(func, df_split)
    df = pd.concat(map_values)
    pool.close()
    pool.join()
    return df

def check_children(syn, data_parent_id, filename):
    """
    Utility function to check if file is already available
    If file is available, get all the recordIds and all the file
    Args: 
        syn                       = syn object           
        data_parent_id  (string)  = the synId parent folder
        filename (string)         = the filename being searched
    Returns: 
        Previously stored dataframe that has the same filename parameter
    """
    prev_stored_data = pd.DataFrame({"recordId":[]})
    for children in syn.getChildren(parent = data_parent_id):
            if children["name"] == filename:
                prev_stored_data_id = children["id"]
                prev_stored_data = get_file_entity(syn, prev_stored_data_id)
    return prev_stored_data


def get_git_used_script_url(path_to_github_token, 
                            proj_repo_name, script_name):
    """
    Utility function to get running script based on current git HEAD
    of where the script is being executed.

    Args:
        path_to_github_token (dtype:string) = filepath to github token
        repo_name            (dtype:string) = name of project repo
        script_name          (dtype:string) = name of executed script

    Returns:
        RType: String
        Returns a string of URL to github on executed script
    """
    try:
        with open(os.path.expanduser("~/git_token.txt"), "r+") as f:
            token = f.read().strip()
        g = Github(token)
        user_name     = g.get_user().login
        head_sha      = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
        relative_path = subprocess.check_output(["git", "rev-parse", "--show-prefix"]).decode('ascii').strip()
    except:
        raise("some error message, TODO")
    else:
        script_url = "https://github.com/%s/%s/blob/%s/%s%s"%(user_name, 
                                                            proj_repo_name, 
                                                            head_sha, 
                                                            relative_path, 
                                                            script_name)
    return script_url



    
