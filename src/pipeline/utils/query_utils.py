## future library imports ## 
from __future__ import unicode_literals
from __future__ import print_function

## standard library imports ##
import sys
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
        5.) Annotate any empty filepaths as "#ERROR"
    
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
                        (MPOWER_V1, MS_ACTIVE, MPOWER_V2, PASSIVE)")
    
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
    data = data.fillna("#ERROR") 
    cols = [feat for feat in data.columns if "path_id" not in feat]
    return data[cols]

def get_sensor_data_from_filepath(self, filepath, sensor): 
        """
        Utility Function to get sensor data given a filepath, and sensor type
        will adjust to different table entity versions accordingly by 
        extracting specific keys in json patterns. 
        
        Note: Empty filepaths, Empty Dataframes will be annotated with "#ERROR"

        Args: 
            filepath (type: string): string of filepath
            sensor   (type: string): the sensor type (userAcceleration, 
                                                    acceleration with gravity, 
                                                    gyroscope etc. from time series)

        Returns:
            Rtype: pd.DataFrame
            Return a formatted version of the dataframe that contains an index of time-index dataframe (timestamp), 
            and columns of time differences in seconds, and sensor measurement in x, y, z coordinate from the filepaths
        """
        ## if empty filepaths return it back ##
        if not (isinstance(filepath, str) and (filepath != "#ERROR")):
            return "#ERROR"
        
        ## open filepath ##
        with open(filepath) as f:
            json_data = f.read()
            data = pd.DataFrame(json.loads(json_data))

        ## return accelerometer data back if empty ##
        if data.shape[0] == 0: 
            return "#ERROR"
        
        ## get data from mpowerV2 column patterns ##
        if ("sensorType" in data.columns):
            try:
                data = data[data["sensorType"] == sensor]
            except:
                return "#ERROR"
        
        ## get data from mpowerV1 column patterns ##
        else:
            try:
                data = data[["timestamp", sensor]]
                data["x"] = data[sensor].apply(lambda x: x["x"])
                data["y"] = data[sensor].apply(lambda x: x["y"])
                data["z"] = data[sensor].apply(lambda x: x["z"])
                data = data.drop([sensor], axis = 1)
            except:
                return "#ERROR"
        
        ## format dataframe to dateTimeIndex, td, x, y, z, AA ## 
        return format_time_series_data(data)
    

def format_time_series_data(data):
    """
    Utility function to clean accelerometer data to a desirable format 
    required by the PDKIT package.
    
    Args: 
        data(type: pd.DataFrame): pandas dataframe of time series
    
    Returns:
        RType: pd.DataFrame
        Returns an indexed datetimeindex in seconds, time differences in seconds from the start of the test (float), 
        x (float), y (float), z (float), AA (float) 
    """
    if data.shape[0] == 0:
        raise Exception("Empty DataFrame")
    data = data.dropna(subset = ["x", "y", "z"])
    date_series = pd.to_datetime(data["timestamp"], unit = "s")
    data["td"] = (date_series - date_series.iloc[0]).apply(lambda x: x.total_seconds())
    data["time"] = data["td"]
    data = data.set_index("time")
    data.index = pd.to_datetime(data.index, unit = "s")
    data["AA"] = np.sqrt(data["x"]**2 + data["y"]**2 + data["z"]**2)
    data = data.sort_index()
    return data[["td","x","y","z","AA"]] 
    
    
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


def generate_demographic_info(syn, data):
    """
    Utility function for gathering demographic data
    """

    
    DEMO_DATA_V1 = "syn10371840"
    DEMO_DATA_V2 = "syn15673379"
    
    ## demographics on mpower version 1 ##
    demo_data_v1 = syn.tableQuery("SELECT age, healthCode, \
                                inferred_diagnosis as PD,  \
                                gender FROM {} \
                                where dataGroups NOT LIKE '%test_user%'".format(DEMO_DATA_V1)).asDataFrame()
    demo_data_v1 = demo_data_v1[(demo_data_v1["gender"] == "Female") | (demo_data_v1["gender"] == "Male")]
    demo_data_v1 = demo_data_v1.dropna(subset = ["PD"], thresh = 1)                     ## drop if no diagnosis
    demo_data_v1["PD"] = demo_data_v1["PD"].map({True :1.0, False:0.0})                 ## encode as numeric binary
    demo_data_v1["age"] = demo_data_v1["age"].apply(lambda x: float(x))  
    demo_data_v1["gender"] = demo_data_v1["gender"].apply(lambda x: x.lower())
    
    ## demographics on mpower version 2 ##
    demo_data_v2 = syn.tableQuery("SELECT birthYear, createdOn, healthCode, \
                                    diagnosis as PD, sex as gender FROM {} \
                                    where dataGroups NOT LIKE '%test_user%'".format(DEMO_DATA_V2)).asDataFrame()
    demo_data_v2        = demo_data_v2[(demo_data_v2["gender"] == "male") | (demo_data_v2["gender"] == "female")]
    demo_data_v2        = demo_data_v2[demo_data_v2["PD"] != "no_answer"]               
    demo_data_v2["PD"]  = demo_data_v2["PD"].map({"parkinsons":1, "control":0})
    demo_data_v2["birthYear"] = demo_data_v2[demo_data_v2["birthYear"].apply(lambda x: True if x>=0 else False)]
    demo_data_v2["age"] = pd.to_datetime(demo_data_v2["createdOn"], unit = "ms").dt.year - demo_data_v2["birthYear"] 
    
    
    demo_data = pd.concat([demo_data_v1, demo_data_v2]).reset_index(drop = True)
    
    ## check integrity of data ##
    ## check if multiple input of PD ##
    demo_data = pd.merge(demo_data, 
         (demo_data.groupby("healthCode")\
          .nunique()["PD"] >= 2)\
            .reset_index()\
                .rename({"PD":"has_double_PD_entry"}, axis = 1),
         on = "healthCode", 
         how = "left")
    demo_data = demo_data[demo_data["has_double_PD_entry"] == False]
    
    ## realistic age range ##
    demo_data = demo_data[(demo_data["age"] <= 110) & (demo_data["age"] >= 10)]
    demo_data = demo_data.drop(["birthYear","createdOn", "has_double_PD_entry"], axis = 1)  
    
    data = pd.merge(data, demo_data, how = "inner", on = "healthCode")
    return data



    
