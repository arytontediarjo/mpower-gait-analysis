"""
Author: Sage Bionetworks

Script to gather all Demographics Data from all gait data from Synapse Table
"""

# import future libraries
from __future__ import print_function
from __future__ import unicode_literals

# import standard libraries
import time
import os
import pandas as pd
import numpy as np

# import external libraries
import synapseclient as sc

# import project modules
from utils import query_utils as query

# global variables
DATA_DICT = {
    "DEMOGRAPHICS": {
        "MPOWER_V1": "syn10371840",
        "MPOWER_V2": "syn15673379",
        "ELEVATE_MS_DEMO": "syn10295288",
        "ELEVATE_MS_PROF": "syn10235463"},
    "OUTPUT_INFO": {
        "FILENAME": "gait_demographics.csv",
        "PARENT_SYN_ID": "syn21537423",
        "PROJ_REPO": "mpower-gait-analysis",
        "TOKEN_PATH": "~/git_token.txt"}
}


def generate_gait_demographic(syn):
    """
    Function to generate healthcode demographic informations
    from Demographic and Profiles synapse table.
    Takes in a dataframe containing healthcode,
    and join the table and compiled demographic
    data by their healthcodes.

    Cleaning process:
        1. Annotate controls, PD, MS in a column called class
        2. Filter healthCode if gender, inferred diagnosis is NULL
        3. If age is recorded as birthYear, age
            will be based on current year - birthYear
        4. healthCodes that has double PD status entry
            will be dropped from the dataframe
        5. Age is subsetted between 18-120 years old
        6. Aggregation of records for each healthcodes will be based on
        number of unique record entries, other metadata features will be
        aggregated based on most frequent occurences

    Args:
        syn                : synapseclient object
        data (pd.DataFrame): pandas dataframe

    Returns:
        RType: pd.DataFrame
        Returns a dataframe of unique healthcode and
        its corresponding metadata features
    """

    # demographics on mpower V1
    demo_data_v1 = syn.tableQuery(
        "SELECT age, healthCode, \
        inferred_diagnosis as PD, gender \
        FROM {} where dataGroups \
        NOT LIKE '%test_user%'"
        .format(DATA_DICT["DEMOGRAPHICS"]["MPOWER_V1"]))\
        .asDataFrame()
    demo_data_v1 = demo_data_v1\
        .dropna(subset=["PD"], thresh=1)
    demo_data_v1["class"] = demo_data_v1["PD"]\
        .map({True: "PD", False: "control"})

    # demographics on ElevateMS
    demo_data_ems = syn.tableQuery(
        "SELECT healthCode, dataGroups as MS,\
        'gender.json.answer' as gender from {}\
        where dataGroups NOT LIKE '%test_user%'"
        .format(DATA_DICT["DEMOGRAPHICS"]["ELEVATE_MS_DEMO"]))\
        .asDataFrame()
    profile_data_ems = syn.tableQuery(
        "SELECT healthCode as healthCode, \
        'demographics.age' as age from {}"
        .format(DATA_DICT["DEMOGRAPHICS"]["ELEVATE_MS_PROF"]))\
        .asDataFrame()
    demo_data_ems = pd.merge(
        demo_data_ems, profile_data_ems,
        how="inner", on="healthCode")
    demo_data_ems["class"] = demo_data_ems["MS"].map(
        {"ms_patient": "MS", "control": "control"})

    # demographics on mpower V2
    demo_data_v2 = syn.tableQuery(
        "SELECT birthYear, createdOn, healthCode, \
        diagnosis as PD, sex as gender FROM {} \
        where dataGroups NOT LIKE '%test_user%'"
        .format(DATA_DICT["DEMOGRAPHICS"]["MPOWER_V2"])).asDataFrame()
    demo_data_v2 = demo_data_v2[demo_data_v2["PD"] != "no_answer"]
    demo_data_v2["class"] = demo_data_v2["PD"]\
        .map({"parkinsons": "PD", "control": "control"})
    demo_data_v2["birthYear"] = demo_data_v2[demo_data_v2["birthYear"]
                                             .apply(lambda x: True if x >= 0
                                                    else False)]
    demo_data_v2["age"] =\
        pd.to_datetime(demo_data_v2["createdOn"],
                       unit="ms").dt.year - demo_data_v2["birthYear"]

    # concatenate all demographic data
    demo_data = pd.concat(
        [demo_data_v1, demo_data_v2, demo_data_ems], sort=False)\
        .reset_index(drop=True)

    # filter gender
    demo_data["gender"] = demo_data["gender"].str.lower()
    demo_data = demo_data[(demo_data["gender"] == "female")
                          | (demo_data["gender"] == "male")]

    # filter age
    demo_data["age"] = demo_data["age"].apply(lambda x: float(x))
    demo_data = demo_data[(demo_data["age"] <= 120) & (demo_data["age"] >= 18)]
    demo_data = demo_data[~demo_data["age"].isin([np.inf, -np.inf])]
    demo_data = demo_data.sort_values(by="age", ascending=False)

    # check if multiple input of any class
    demo_data = pd.merge(demo_data,
                         (demo_data.groupby("healthCode")
                          .nunique()["class"] >= 2)
                         .reset_index()
                         .rename({"class": "has_double_class_entry"}, axis=1),
                         on="healthCode",
                         how="left")
    demo_data = demo_data.drop(
        ["PD", "MS", "birthYear",
         "createdOn", "has_double_class_entry"], axis=1)
    demo_data = demo_data.drop_duplicates(
        'healthCode', keep="first").reset_index(drop=True)
    return demo_data


def main():
    """
    Main Function
    Entry point for the script
    Note: Passive gait data will be separated from active gait data
          as we dont want to combine both in analysis
    """

    # retrieve synapse credential through config
    path = os.path.join(os.getenv("HOME"),
                        ".synapseConfig")
    syn = sc.Synapse(configPath=path)
    syn.login(os.getenv("syn_username"),
              os.getenv("syn_password"),
              rememberMe=True)

    # process metadata from synapse table
    metadata = generate_gait_demographic(syn)

    # get this script git blob URL
    used_script_url = query.get_git_used_script_url(
        path_to_github_token=DATA_DICT["OUTPUT_INFO"]["TOKEN_PATH"],
        proj_repo_name=DATA_DICT["OUTPUT_INFO"]["PROJ_REPO"],
        script_name=__file__)

    # save data to synapse
    query.save_data_to_synapse(
        syn=syn,
        data=metadata,
        source_table_id=[dataframe
                         for key, dataframe
                         in DATA_DICT["DEMOGRAPHICS"].items()],
        used_script=used_script_url,
        output_filename=DATA_DICT["OUTPUT_INFO"]["FILENAME"],
        data_parent_id=DATA_DICT["OUTPUT_INFO"]["PARENT_SYN_ID"])


if __name__ == '__main__':
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
