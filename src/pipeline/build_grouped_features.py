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
data_dict = {"GAIT_DATA": {"synId": "syn21575055"},
            "GAIT_METADATA": {"synId": "syn21590710"},
            "OUTPUT_INFO"      : {"parent_folder_synId"    : "syn21592268",
                                    "proj_repo_name"       : "mpower-gait-analysis",
                                    "path_to_github_token" : "~/git_token.txt"}
    }

syn = sc.login()

def iqr(x):
    """
    Function for getting IQR value
    """
    return q75(x) - q25(x)
def q25(x):
    """
    Function for getting first quantile
    """
    return x.quantile(0.25)
def q75(x):
    """
    Function for getting third quantile
    """
    return x.quantile(0.75)
def valrange(x):
    """
    Function for getting the value range
    """
    return x.max() - x.min()
def kurtosis(x):
    """
    Function to retrieve kurtosis 
    """
    return x.kurt()
def skew(x):
    """
    Function to retrieve skewness 
    """
    return x.skew()

def groupby_wrapper(data, group, exclude_columns):
    """
    Wrapper function to wrap feature data
    into several aggregation function 
    
    Args:
        data (dtype: pd.Dataframe): featurized data
        group (dtype: string)     : which group to aggregate
    Returns:
        Rtype: pd.Dataframe
        Returns grouped healthcodes features
    """

    features = [feat for feat in data.columns if feat not in exclude_columns]    
    data =  data[features].groupby(group).agg([np.max, 
                                    np.median, 
                                    np.mean,
                                    q25, q75, 
                                    valrange, iqr])
    feature_cols = []
    for feat, agg in data.columns:
        feature_cols_name = "{}.{}".format(agg, feat)
        feature_cols.append(feature_cols_name)
    data.columns = feature_cols
    return data

def main():
    gait_data = query.get_file_entity(syn = syn, 
                                    synid = data_dict["GAIT_DATA"]["synId"])

    gait_metadata = query.get_file_entity(syn = syn, 
                                        synid = data_dict["GAIT_METADATA"]["synId"])

    # ## remove this later ##
    # gait_data = gait_data.drop(['gait_features', 'gait_json_filepath', "window"], axis = 1)

    # gait_data = gait_data.replace(np.inf, np.nan)

    metadata = ['appVersion', 'createdOn',
                'phoneInfo', 'recordId', 
                'table_version', 'test_type'] 

    output_mapping = {"walking_active_data" : gait_data[(gait_data["table_version"] != "MPOWER_PASSIVE") \
                                                            & (gait_data["test_type"] == "walking")],
                      "balance_active_data" : gait_data[(gait_data["table_version"] != "MPOWER_PASSIVE") \
                                                            & (gait_data["test_type"] != "walking")],
                      "passive_data": gait_data[(gait_data["table_version"] == "MPOWER_PASSIVE") \
                                                            & (gait_data["test_type"] == "walking")]}

    used_script_url = query.get_git_used_script_url(path_to_github_token = data_dict["OUTPUT_INFO"]["path_to_github_token"],
                                                    proj_repo_name       = data_dict["OUTPUT_INFO"]["proj_repo_name"],
                                                    script_name          = __file__)

    for data_name, dataframe in output_mapping.items():
        grouped_data = groupby_wrapper(dataframe, "healthCode", metadata)
        grouped_data = pd.merge(grouped_data, gait_metadata, on = "healthCode", how = "inner")
        query.save_data_to_synapse(syn = syn,
                                    data = grouped_data,
                                    used_script = used_script_url,
                                    output_filename = "grouped_%s.csv"%data_name,
                                    data_parent_id = "syn21537421")

if __name__ ==  '__main__': 
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))


