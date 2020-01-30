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
    Query synapse walking table entity 
    parameters:  
    syn         : synapse object,             
    table_id    : id of table entity,
    version     : version number (args (string) = ["MPOWER_V1", "MPOWER_V2", "MS_ACTIVE", "PASSIVE"])
    healthcodes : list or array of healthcodes
    recordIDs   : list or of recordIds
    
    returns: a dataframe of recordIds and their respective metadata, alongside their filehandleids and filepaths
             empty filepath will be annotated as "#ERROR" on the dataframe
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
        Function to get sensor data given a filepath, and sensor type
        will adjust to different table entity versions accordingly by 
        extracting specific keys in json patterns. 
        
        Note: Empty filepaths, Empty Dataframes will be annotated with "#ERROR"

        Args: 
            filepath (type: string): string of filepath
            sensor   (type: string): the sensor type (userAcceleration, 
                                                    acceleration with gravity, 
                                                    gyroscope etc. from time series)

        return a formatted version of the dataframe that contains a time-index dataframe (timestamp), 
        time differences , (x, y, z, AA) user acceleration (non-g)
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
        if data.shape[0] != 0:
            return format_time_series_data(data)
        else:
            return "#ERROR"
    

def format_time_series_data(data):
    """
    Generalized function to clean accelerometer data to a desirable format 
    parameter: 
    `data`: pandas dataframe of time series
    returns index (datetimeindex), td (float64), 
            x (float64), y (float64), z (float64),
            AA (float64) dataframe    
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
                        used_script = None,
                        source_table_id = None,
                        remove = True): 
    """
    Function to save data to synapse given a parent id, used script, 
    and source table where the query was sourced
    params: 
    `syn`              = synapse object        
    `data`             = tabular data, script or notebook 
    `output_filename`  = the name of the output file 
    `data_parent_id`   = the parent synid where data will be stored 
    `used_script`      = git repo url that produces this data (if available)
    `source_table_id`  = list of source of where this data is produced (if available) 
    `remove`           = remove data after saving, generally used for csv data 

    returns stored file entity in Synapse Database
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
    Function to normalize column that conatins dictionaries into separate columns
    in the dataframe
    parameter: 
    `data`    : pandas DataFrame       
    `features` : list of dict target features for normalization 
    returns a normalized dataframe with column containing dictionary normalized
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
    Function to normalize list of dicts into dataframe of rows
    parameter:
        `data`: pandas DataFrame
        `features`: a list of features for normalization to rows
    return a normalized dataframe with new rows from normalize list of dicts
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
    Get data (csv,tsv) file entity and turn it into pandas csv
    returns pandas dataframe 
    parameters:
    `syn`: a syn object
    `synid`: syn id of file entity
    returns pandas dataframe
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
    Function for parallelizing pandas dataframe processing
    parameter: 
    `df`               = pandas dataframe         
    `func`             = wrapper function for data processing
    `no_of_processors` = number of processors to transform the data
    `chunksize`        = number of partition 
    return: featurized dataframes
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
    Function to check if file is already available
    if file is available, get all the recordIds and all the file
    parameter: 
    `syn` = syn object           
    `data_parent_id` = the parent folder
    `output_filename` = the filename
    returns previously stored dataframe that has the same filename
    """
    prev_stored_data = pd.DataFrame({"recordId":[]})
    for children in syn.getChildren(parent = data_parent_id):
            if children["name"] == filename:
                prev_stored_data_id = children["id"]
                prev_stored_data = get_file_entity(syn, prev_stored_data_id)
    return prev_stored_data


def generate_demographic_info(syn, data):
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



    
