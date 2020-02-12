"""
Author: Sage Bionetworks

Script to gather all Demographics Data from all gait data from Synapse Table
"""

## import future libraries ## 
from __future__ import print_function
from __future__ import unicode_literals

## import standard libraries ## 
import time
import pandas as pd
import numpy as np

## import external libraries ##
import synapseclient as sc

## import project modules
from utils import query_utils as query

## global variables ## 
data_dict = {"GAIT_FEATURE_DATA": {"synId": "syn21575055"},
            "DEMO_DATA_V1"      : {"synId": "syn10371840"},
             "DEMO_DATA_V2"     : {"synId": "syn15673379"},
             "DEMO_DATA_EMS"    : {"synId": "syn10295288"},
             "PROFILE_DATA_EMS" : {"synId": "syn10235463"},
             "OUTPUT_INFO"      : {"active_metadata_filename"  : "active_gait_metadata.csv",
                                    "passive_metadata_filename" : "passive_gait_metadata.csv",
                                    "parent_folder_synId"       : "syn21537423",
                                    "proj_repo_name"            : "mpower-gait-analysis",
                                    "path_to_github_token"      : "~/git_token.txt"}
    }

syn = sc.login()

## helper functions ## 
def annot_phone(params):
    """
    Function to have lesser phone information choices

    Args:
        params (type: string): raw phone information
    
    Returns:
        Rtype: String
        Returns an annotated dataset with lesser choice of phone information
    """
    if ";" in params:
        params = params.split(";")[0]
    if ("iPhone 6+" in params) or ("iPhone 6 Plus" in params):
        return "iPhone 6+"
    elif ("Unknown" in params) or ("iPad" in params) or ("iPod" in params):
        return "Other iPhone"
    elif ("iPhone 5" in params) or ("iPhone5" in params):
        return "iPhone 5"
    elif ("iPhone8" in params) or ("iPhone 8" in params):
        return "iPhone 8"
    elif ("iPhone9" in params) or ("iPhone 9" in params):
        return "iPhone 9"
    elif ("iPhone X" in params) or ("iPhone10" in params):
        return "iPhone X"
    return params

def generate_demographic_info(syn, feature_data):
    """
    Function to generate unique healthcode demographic informations from Demographic and Profiles synapse table.
    Takes in a dataframe containing healthcode, and join the table and compiled demographic 
    data by their healthcodes.

    Cleaning process:
        >> Annotate controls, PD, MS in a column called class
        >> Filter healthCode if gender, inferred diagnosis is NULL
        >> Age, if recorded as birthYear will be based on current year - birthYear
        >> healthCodes that has double PD status entry will be dropped from the dataframe
        >> Age is subsetted between 18-120 years old
        >> Aggregation of records for each healthcodes will be based on number of unique record entries,
            other metadata features will be aggregated based on most frequent occurences

    Args:
        syn                : synapse object
        data (pd.DataFrame): a pandas dataframe object 
    
    Returns:
        RType: pd.DataFrame
        Returns a dataframe of unique healthcode and its corresponding metadata features
    """

    ## demographics on mpower V1 ##
    demo_data_v1 = syn.tableQuery("SELECT age, healthCode, inferred_diagnosis as PD, gender \
                                 FROM {} where dataGroups NOT LIKE '%test_user%'"\
                                .format(data_dict["DEMO_DATA_V1"]["synId"])).asDataFrame()
    demo_data_v1 = demo_data_v1.dropna(subset = ["PD"], thresh = 1)                               
    demo_data_v1["class"] = demo_data_v1["PD"].map({True :"PD", False:"control"})


    ## demographics on ElevateMS ##
    demo_data_ems = syn.tableQuery("SELECT healthCode, dataGroups as MS,\
                                'gender.json.answer' as gender from {}\
                                where dataGroups NOT LIKE '%test_user%'"\
                                .format(data_dict["DEMO_DATA_EMS"]["synId"])).asDataFrame()
    profile_data_ems = syn.tableQuery("SELECT healthCode as healthCode, \
                                    'demographics.age' as age from {}"\
                                    .format(data_dict["PROFILE_DATA_EMS"]["synId"])).asDataFrame()
    demo_data_ems = pd.merge(demo_data_ems, profile_data_ems, how = "inner", on = "healthCode")
    demo_data_ems["class"] = demo_data_ems["MS"].map({"ms_patient":"MS", "control":"control"})


    ## demographics on mpower V2 ##
    demo_data_v2 = syn.tableQuery("SELECT birthYear, createdOn, healthCode, \
                                    diagnosis as PD, sex as gender FROM {} \
                                    where dataGroups NOT LIKE '%test_user%'"\
                                    .format(data_dict["DEMO_DATA_V2"]["synId"])).asDataFrame()
    demo_data_v2              = demo_data_v2[demo_data_v2["PD"] != "no_answer"]               
    demo_data_v2["class"]     = demo_data_v2["PD"].map({"parkinsons":"PD", "control":"control"})
    demo_data_v2["birthYear"] = demo_data_v2[demo_data_v2["birthYear"].apply(lambda x: True if x>=0 else False)]
    demo_data_v2["age"]       = pd.to_datetime(demo_data_v2["createdOn"], unit = "ms").dt.year - demo_data_v2["birthYear"]

    ## concatenate all demographic data
    demo_data = pd.concat([demo_data_v1, demo_data_v2, demo_data_ems], sort = False)\
                            .reset_index(drop = True)

    ## filter gender ## 
    demo_data["gender"] = demo_data["gender"].str.lower()
    demo_data = demo_data[(demo_data["gender"] == "female") | (demo_data["gender"] == "male")]

    ## filter age ##
    demo_data["age"] = demo_data["age"].apply(lambda x: float(x))
    demo_data        = demo_data[(demo_data["age"] <= 120) & (demo_data["age"] >= 18)]
    demo_data        = demo_data[~demo_data["age"].isin([np.inf, -np.inf])]
    demo_data        = demo_data.sort_values(by = "age", ascending = False)


    ## filter double entry ##
    ## check if multiple input of any class ##
    demo_data = pd.merge(demo_data, 
        (demo_data.groupby("healthCode")\
        .nunique()["class"] >= 2)\
            .reset_index()\
                .rename({"class":"has_double_class_entry"}, axis = 1),
        on = "healthCode", 
        how = "left")
    demo_data = demo_data.drop(["PD", "MS", "birthYear", "createdOn", "has_double_class_entry"], axis = 1)
    demo_data = demo_data.drop_duplicates('healthCode', keep = "first").reset_index(drop = True)

    ## merge dataframe with demographic data ##
    ## synId
    data         = pd.merge(feature_data, demo_data, how = "inner", on = "healthCode")
    
    ## aggregation of metadata features ##
    data= data[["recordId", "healthCode", "phoneInfo", "age", "gender", "class", "table_version"]].groupby(["healthCode"])\
                .agg({"recordId": pd.Series.nunique,
                     "phoneInfo": pd.Series.mode,
                     "class": pd.Series.mode,
                     "gender": pd.Series.mode,
                     "age": pd.Series.mode,
                     "table_version": pd.Series.mode})
    data = data.rename({"recordId":"nrecords"}, axis = 1)
    
    ## annotate phone into simpler categories ##
    data["phoneInfo"] = data["phoneInfo"].apply(lambda x: x[0] if not isinstance(x,str) else x)
    data["phoneInfo"] = data["phoneInfo"].apply(annot_phone)
    return data

def main():
    """
    Main Function
    Entry point for the script
    Note: Passive gait data will be separated from active gait data
          as we dont want to combine both in analysis
    """
    ## get featurized gait data ##
    gait_data    = query.get_file_entity(syn = syn, synid = data_dict["GAIT_FEATURE_DATA"]["synId"])
    
    ## filter error data, and separate active and passive data ##
    gait_data    = gait_data[gait_data["error_type"].isnull()]
    active_data  = gait_data[gait_data["table_version"] != "MPOWER_PASSIVE"]
    passive_data = gait_data[gait_data["table_version"] == "MPOWER_PASSIVE"]
    
    output_mapping = {"active_metadata" : generate_demographic_info(syn, active_data), 
                      "passive_metadata" : generate_demographic_info(syn, passive_data)}
    
    ## get this script URL ## 
    used_script_url = query.get_git_used_script_url(path_to_github_token = data_dict["OUTPUT_INFO"]["path_to_github_token"],
                                                    proj_repo_name       = data_dict["OUTPUT_INFO"]["proj_repo_name"],
                                                    script_name          = __file__)

    ## save active data to synapse ##
    for data_name, dataframe in output_mapping.items():
        query.save_data_to_synapse(syn = syn,
                                    data = dataframe,
                                    source_table_id = [values["synId"] for key, values in data_dict.items() if key != "OUTPUT_INFO"],
                                    used_script = used_script_url,  
                                    output_filename = "%s.csv"%data_name,
                                    data_parent_id  = data_dict["OUTPUT_INFO"]["parent_folder_synId"])
    
if __name__ ==  '__main__': 
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
