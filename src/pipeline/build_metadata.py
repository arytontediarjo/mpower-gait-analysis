from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import sys
sys.path.append("../../")
import synapseclient as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from src.pipeline.utils import query_utils as query
import synapseclient as sc
from sklearn import metrics
import time
import argparse
import multiprocessing
warnings.simplefilter("ignore")

DEMO_DATA_V1  = "syn10371840"
DEMO_DATA_V2  = "syn15673379"
DEMO_DATA_EMS = "syn10295288"
syn = sc.login()

## helper functions ## 
def annot_phone(params):
    """
    Function to annotate phone types
    parameter:
    `params`: raw phone type string
    
    returns an annotated dataset with lesser choice of phonetypes
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

def generate_demographic_info(syn, data):
    ## demographics on mpower version 1 ##
    demo_data_v1 = syn.tableQuery("SELECT age, healthCode, \
                                inferred_diagnosis as PD,  \
                                gender FROM {}\
                                where dataGroups NOT LIKE '%test_user%'".format(DEMO_DATA_V1)).asDataFrame()
    demo_data_v1 = demo_data_v1[(demo_data_v1["gender"] == "Female") | (demo_data_v1["gender"] == "Male")]
    demo_data_v1 = demo_data_v1.dropna(subset = ["PD"], thresh = 1)                     ## drop if no diagnosis
    demo_data_v1["class"] = demo_data_v1["PD"].map({True :"PD", False:"control"})                 ## encode as numeric binary
    demo_data_v1["age"] = demo_data_v1["age"].apply(lambda x: float(x))  

    ## demographics on ElevateMS ##
    demo_data_ems = syn.tableQuery("SELECT healthCode, dataGroups as MS,\
                                'gender.json.answer' as gender from {}\
                                where dataGroups NOT LIKE '%test_user%'".format(DEMO_DATA_EMS)).asDataFrame()
    demo_data_ems = demo_data_ems[(demo_data_ems["gender"] == "Male") | (demo_data_ems["gender"] == "Female")]
    demo_data_ems["class"] = demo_data_ems["MS"].map({"ms_patient":"MS", "control":"control"})
    demo_data_ems["age"]  = 0
    
    ## demographics on mpower version 2 ##
    demo_data_v2 = syn.tableQuery("SELECT birthYear, createdOn, healthCode, \
                                    diagnosis as PD, sex as gender FROM {} \
                                    where dataGroups NOT LIKE '%test_user%'".format(DEMO_DATA_V2)).asDataFrame()
    demo_data_v2        = demo_data_v2[(demo_data_v2["gender"] == "male") | (demo_data_v2["gender"] == "female")]
    demo_data_v2        = demo_data_v2[demo_data_v2["PD"] != "no_answer"]               
    demo_data_v2["class"]  = demo_data_v2["PD"].map({"parkinsons":"PD", "control":"control"})
    demo_data_v2["birthYear"] = demo_data_v2[demo_data_v2["birthYear"].apply(lambda x: True if x>=0 else False)]
    demo_data_v2["age"] = pd.to_datetime(demo_data_v2["createdOn"], unit = "ms").dt.year - demo_data_v2["birthYear"] 
    
    
    demo_data = pd.concat([demo_data_v1, demo_data_v2, demo_data_ems]).reset_index(drop = True)
    
    ## realistic age range ##
    demo_data = demo_data[(demo_data["age"] <= 120) & (demo_data["age"] >= 0)]

    ##lower case gender ## 
    demo_data["gender"] = demo_data["gender"].apply(lambda x: x.lower())

    ## check if multiple input of PD ##
    demo_data = pd.merge(demo_data, 
         (demo_data.groupby("healthCode")\
          .nunique()["class"] >= 2)\
            .reset_index()\
                .rename({"class":"has_double_class_entry"}, axis = 1),
         on = "healthCode", 
         how = "left")
    demo_data = demo_data[demo_data["has_double_class_entry"] == False]
    demo_data = demo_data.drop(["birthYear","createdOn", "PD", "MS", "has_double_class_entry"], axis = 1)  
    
    data = pd.merge(data, demo_data, how = "inner", on = "healthCode")
    
    data= data[["recordId", "healthCode", "phoneInfo", "age", "gender", "class", "version"]].groupby(["healthCode"])\
                .agg({"recordId": pd.Series.nunique,
                     "phoneInfo": pd.Series.mode,
                     "class": pd.Series.mode,
                     "gender": pd.Series.mode,
                     "age": pd.Series.mode,
                     "version": pd.Series.mode})
    data = data.rename({"recordId":"nrecords"}, axis = 1)
    
    ## integrity checking ##
    data["phoneInfo"] = data["phoneInfo"].apply(lambda x: x[0] if not isinstance(x,str) else x)
    data = data[data["age"].apply(lambda x: isinstance(x,(int,float)))]
    data.ix[data["class"]  == "MS", ["age"]] = "#ERROR"
    data["phoneInfo"] = data["phoneInfo"].apply(annot_phone)
    return data

def main():
    gait_data    = query.get_file_entity(syn = syn, synid = "syn21542870")
    active_data  = gait_data[gait_data["version"] != "mpower_passive"]
    passive_data = gait_data[gait_data["version"] == "mpower_passive"]
    active_metadata = generate_demographic_info(syn, active_data)
    passive_metadata = generate_demographic_info(syn, passive_data)
    
    ## save data to synapse ##
    query.save_data_to_synapse(syn = syn,
                                data = active_metadata,
                                output_filename = "active_gait_user_metadata.csv",
                                data_parent_id = "syn21537423")
    ## save data to synapse ##
    query.save_data_to_synapse(syn = syn,
                                data = passive_metadata,
                                output_filename = "passive_gait_user_metadata.csv",
                                data_parent_id = "syn21537423")

if __name__ ==  '__main__': 
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))



