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
from utils import gait_features_utils as gf_utils

## global variables ## 
GAIT_DATA     = "syn21542870"
ROTATION_DATA = "syn21542869"
MATCHED_DATA    = "syn21547110"
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

def separate_data_by_axis(data, axis):
    """
    Function to separate data by its axial coordinates
    and annotate column names by its axial coordinate
    example: <axis>.<name_of_feature>
    
    Args:
        data (type: pd.DataFrame): pandas dataframe of features, with ROWS of different axis
        axis (type: string)      : axial coordinates
    """
    axis_name = [feat for feat in data.columns if "axis" in feat][0]
    data = data[data[axis_name] == axis].reset_index(drop = True)
    data.columns = ["{}.{}".format(axis, cols) if "." in cols else cols for cols in data.columns]
    return data

def group_features(data, coord_list, filtered = False):
    """
    Function to group healthcodes by several aggregation computation (max, median, mean, etc)

    Args:
        data (type: pd.DataFrame): pandas dataframe that consists of columns of recordIds, healthCodes and features
        axis (tupe: string)      : axial coordinates
    
    Returns:
        RType: pd.DataFrame
        A grouped healthcode feature dataframe with aggregated features
    """

    # gaitfeatures = gf_utils.GaitFeaturize()
    data = data[[feat for feat in data.columns if ("." in feat) \
                 or ("healthCode" in feat) or ('recordId' in feat)]]
    data_dict = {}
    for coordinate in coord_list:
        axial_data = separate_data_by_axis(data, coordinate)

        #TODO: check if filtering will be useful for data # 
        # if filtered:
        #     feat = [feat for feat in axial_data.columns if (feat == "%s.walking.steps"%coordinate)\
        #            or (feat == "%s.rotation.steps"%coordinate)][0]
        #     axial_data = gaitfeatures.annotate_consecutive_zeros(axial_data, feat).drop(["recordId"], axis = 1)
        #     axial_data = axial_data[axial_data["consec_zero_steps_count"] < 15].drop(["consec_zero_steps_count"], axis = 1)
        axial_data = axial_data.groupby("healthCode").agg([np.max, 
                                                   np.median, 
                                                   np.mean,
                                                   q25, q75, valrange, iqr])
        data_dict[coordinate] = axial_data
    data = data_dict[[*data_dict][0]]
    for coordinate in [*data_dict][1:]:
        data = pd.merge(data, data_dict[coordinate], on = "healthCode", how = "inner")
    new_cols = []    
    for feat, agg in data.columns:
        new_cols_name = "{}.{}".format(agg, feat)
        new_cols.append(new_cols_name)
    data.columns = new_cols
    return data

def main():
    """
    Main Function
    """
    gait_data = query.get_file_entity(syn = syn, synid = GAIT_DATA)
    rotation_data = query.get_file_entity(syn = syn, synid = ROTATION_DATA)
    match_data = query.get_file_entity(syn = syn, synid = MATCHED_DATA)
    
    ## get grouped rotation data ##
    df = rotation_data[(rotation_data["table_version"] != "MPOWER_PASSIVE") \
                        & (rotation_data["table_version"] != "ELEVATE_MS") \
                        & (rotation_data["test_type"] == "walking")]
    df = group_features(df, coord_list = ["y"])
    matched_rotation_analysis_data = pd.merge(df, match_data, on = "healthCode", how = "inner")
    query.save_data_to_synapse(syn = syn,
                                data = matched_rotation_analysis_data,
                                output_filename = "grouped_rotation_gait_features.csv",
                                data_parent_id = "syn21537421")
    
    ## get grouped nonrotation data ##
    df = gait_data[(gait_data["table_version"] != "MPOWER_PASSIVE")\
                    & (gait_data["table_version"] != "ELEVATE_MS") \
                    & (gait_data["test_type"] == "walking")]
    df = group_features(df, coord_list = ["x","y","z","AA"])
    matched_walking_analysis_data = pd.merge(df, match_data, on = "healthCode", how = "inner")
    query.save_data_to_synapse(syn = syn,
                                data = matched_walking_analysis_data,
                                output_filename = "grouped_nonrotation_gait_features.csv",
                                data_parent_id = "syn21537421")
    

if __name__ ==  '__main__': 
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))


    
