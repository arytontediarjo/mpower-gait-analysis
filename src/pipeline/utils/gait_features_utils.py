## future liibrary imports ## 
from __future__ import unicode_literals
from __future__ import print_function

## standard library import ## 
import time
import json
import pandas as pd
import numpy as np
from sklearn import metrics

## pdkit imports ##
import pdkit
import pdkit.processor 
from pdkit.utils import (butter_lowpass_filter, 
                         numerical_integration)
## library imports ##
import utils.query_utils as query

class GaitFeaturize:
    
    def __init__(self, 
                rotation_frequency_cutoff   = 2,
                rotation_filter_order       = 2,
                rotation_aucXt_upper_limit  = 2,
                pdkit_gait_frequency_cutoff = 5, 
                pdkit_gait_filter_order     = 4,
                pdkit_gait_delta            = 0.5,
                variance_cutoff             = 1e-3,
                window_size                 = 512,
                step_size                   = 50,
                loco_band                   = [0.5, 3],
                freeze_band                 = [3,8],
                sampling_frequency          = 100
                ):
        self.rotation_frequency_cutoff   = rotation_frequency_cutoff
        self.rotation_filter_order       = rotation_filter_order
        self.rotation_aucXt_upper_limit  = rotation_aucXt_upper_limit
        self.pdkit_gait_filter_order     = pdkit_gait_filter_order
        self.pdkit_gait_frequency_cutoff = pdkit_gait_frequency_cutoff
        self.pdkit_gait_delta            = pdkit_gait_delta
        self.variance_cutoff             = variance_cutoff
        self.window_size                 = window_size
        self.step_size                   = step_size
        self.loco_band                   = loco_band
        self.freeze_band                 = freeze_band
        self.sampling_frequency          = sampling_frequency
        self.gait_processor              = pdkit.processor.Processor(sampling_frequency = self.sampling_frequency)


    def calculate_freeze_index(self, accel_series, sample_rate):
            """
            Modified pdkit FoG freeze index function to be compatible with 
            current source dode
            Args: 
                accel_series (type = pd.Series): pd.Series of acceleration signal in one axis
                sample_rate (type = float64)  : signal sampling rate

            Returns:
                RType: array
                array of [energy of freeze index , locomotor freeze index]
            """
            loco_band   = self.loco_band
            freeze_band = self.freeze_band
            window_size = accel_series.shape[0]
            f_res       = sample_rate / window_size
            f_nr_LBs    = int(loco_band[0] / f_res)
            f_nr_LBe    = int(loco_band[1] / f_res)
            f_nr_FBs    = int(freeze_band[0] / f_res)
            f_nr_FBe    = int(freeze_band[1] / f_res)

            ## normalize series  ## 
            normalized_accel_series = accel_series - np.mean(accel_series)

            ## discrete fast fourier transform ##
            Y = np.fft.fft(normalized_accel_series, int(window_size))
            Pyy = abs(Y*Y) / window_size
            areaLocoBand = numerical_integration( Pyy[f_nr_LBs-1 : f_nr_LBe],   sample_rate)
            areaFreezeBand = numerical_integration( Pyy[f_nr_FBs-1 : f_nr_FBe], sample_rate)
            sumLocoFreeze = areaFreezeBand + areaLocoBand
            freezeIndex = areaFreezeBand / areaLocoBand
            return freezeIndex, sumLocoFreeze



    def get_gait_rotation_info(self, gyro_dataframe, axis = "y"): 
        """
        Function to outputs rotation information (omega, period of rotation) 
        in a dictionary format, given a gyroscope dataframe and its axis orientation

        Args:
            gyro_dataframe (type: pd.DataFrame)               : A gyrosocope dataframe
            axis           (type: str, default = "y")         : Axis of desired rotation orientation,
                                                                default is y axis (pocket-test), 
                                                                unless specified otherwise                                  
        Returns:
            RType: Dictionary

        Return format {"rotation_chunk_1" : {omega: some_value, period: some_value},
                        "rotation_chunk_2": {omega: some_value, period: some_value}}
        """

        ## check if dataframe is valid ## 
        if not isinstance(gyro_dataframe, pd.DataFrame):
            raise Exception("Please parse pandas dataframe into the parameter")

        if gyro_dataframe.shape[0] == 0:
            raise Exception("Dataframe is empty")

        gyro_dataframe[axis] = butter_lowpass_filter(data        = gyro_dataframe[axis],
                                                    sample_rate  = self.sampling_frequency,
                                                    cutoff       = self.rotation_frequency_cutoff, 
                                                    order        = self.rotation_filter_order) 
        zero_crossings = np.where(np.diff(np.sign(gyro_dataframe[axis])))[0]
        start = 0
        num_rotation_window = 0
        rotation_dict = {}
        for crossing in zero_crossings:
            if (gyro_dataframe.shape[0] >= 2) & (start != crossing):
                duration = gyro_dataframe["td"][crossing] - gyro_dataframe["td"][start]
                auc = np.abs(metrics.auc(gyro_dataframe["td"][start : crossing + 1], 
                                         gyro_dataframe[axis][start : crossing + 1]))
                omega = auc/duration
                aucXt = auc * duration

                if aucXt > self.rotation_aucXt_upper_limit:
                    num_rotation_window += 1
                    rotation_dict["rotation_chunk_%s"%num_rotation_window] = ({"omega"     : omega,
                                                                                "aucXt"    : aucXt,
                                                                                "duration" : duration,
                                                                                "period"   : [start,crossing]})
                start = crossing  
        return rotation_dict


    def split_gait_dataframe_to_chunks(self, accel_dataframe, periods):
        """
        Function to chunk dataframe into several segments of rotational and non-rotational sequences into List.
        Each chunk in the list will be in dictionary format containing information of the dataframe 
        and what type of chunk it categorizes as. 

        Args:
            accel_dataframe (type: pd.Dataframe) : An user acceleration dataframe
            periods         (type: List)         : A list of list of periods  ([[start_ix, end_ix], 
                                                                                [start_ix, end_ix]])
        Returns:
            Rtype: List
            Returns a List of dictionary that has keys containing information of what type of chunk,
            and the dataframe itself. 
            Return format:
                [{chunk1 : some_value, dataframe : pd.Dataframe}, 
                 {chunk2  : some_value, dataframe : pd.Dataframe}]
        """

        ## check if dataframe is valid ##
        if not isinstance(accel_dataframe, pd.DataFrame):
            raise Exception("Please parse pandas dataframe into the parameter")

        if accel_dataframe.shape[0] == 0:
            raise Exception("Dataframe is empty")

        ## instantiate initial values ## 
        chunk_list          = []
        pointer             =  0 
        num_rotation_window =  1
        num_walk_window     =  1

        ## check if there is rotation ##
        if len(periods) == 0:
            chunk_list.append({"dataframe": accel_dataframe, 
                               "chunk"    : "walk_chunk_%s"%num_walk_window})
            return chunk_list


        ## edge case: if rotation occurs at zero ##
        if periods[0][0] == pointer:
            chunk_list.append({"dataframe"   : accel_dataframe.iloc[pointer : periods[0][1] + 1],
                               "chunk"       : "rotation_chunk_%s"%num_rotation_window})
            num_rotation_window += 1
            pointer = periods[0][1] + 1
            periods = periods[1:]

        ## middle case ## 
        for index in periods:
            chunk_list.append({"dataframe": accel_dataframe.iloc[pointer : index[0] + 1], 
                               "chunk"    : "walk_chunk_%s"%num_walk_window})
            num_walk_window += 1
            chunk_list.append({"dataframe": accel_dataframe.iloc[index[0] + 1 : index[1] + 1], 
                               "chunk"    : "rotation_chunk_%s"%num_rotation_window})
            num_rotation_window += 1
            pointer = index[1] + 1

        ## edge case, last bit of data ##
        if pointer < accel_dataframe.shape[0]:
            chunk_list.append({"dataframe": accel_dataframe.iloc[pointer : ], 
                               "chunk"    : "walk_chunk_%s"%num_walk_window})
        return chunk_list


    def featurize_gait_dataframe_chunks_by_window(self, list_of_dataframe_dicts, rotation_dict):
        """
        Function to featurize list of dataframe chunks with moving windows,
        and returns list of dictionary with all the pdkit and rotation features (all x, y, z, AA axis) 
        from each moving window
        Notes:
          >> if dataframe chunk is smaller than the window size or is a rotation dataframe chunk
              it will just be treated as one window
          >> if dataframe is bigger than designated window, then featurize dataframe residing
             on that window, with designated step size

        Args:
            list_of_dataframe_dicts (type = List): A List filled with dictionaries containing key-value pair of 
                                                    the dataframe and its chunk type
            rotation_dict           (type = Dict): A Dictionary containing rotation information on each rotation_chunk
                                                >> format is based on function : get_rotation_info(...)
        Returns:
            RType = List
            Returns a list of features from each window, each window features is stored as dictionary
            inside the list
            Return format:
                [{window1_features:...}, {window2_features:...}, {window3_features:...}]
        """
        feature_list = []
        num_window = 1
        ## separate to rotation and non rotation ## 

        for dataframe_dict in list_of_dataframe_dicts:
            window_size     = self.window_size
            step_size       = self.step_size
            jPos            = window_size 
            curr_chunk      = dataframe_dict["chunk"]
            curr_dataframe  = dataframe_dict["dataframe"]
            if curr_dataframe.shape[0] < 2:
                continue
            if "rotation_chunk" in curr_chunk:
                isRotation = True
                rotation_omega = rotation_dict[curr_chunk]["omega"]
            else:
                isRotation = False
                rotation_omega = np.NaN
            if (len(curr_dataframe) < jPos) or isRotation:
                feature_dict = self.gait_featurize(curr_dataframe)
                feature_dict["rotation_omega"] = rotation_omega
                feature_dict["window"]         = "window_%s"%num_window
                feature_list.append(feature_dict)
                num_window += 1
            else:
                while jPos < len(curr_dataframe):
                    jStart     = jPos - window_size
                    subset     = curr_dataframe.iloc[jStart:jPos]
                    feature_dict = self.gait_featurize(subset)
                    feature_dict["rotation_omega"] = rotation_omega
                    feature_dict["window"]         = "window_%s"%num_window
                    feature_list.append(feature_dict)
                    jPos += step_size
                    num_window += 1   
        return feature_list


    def gait_featurize(self, accel_dataframe):
        """
        Function to featurize dataframe using pdkit package (gait).
        Features from pdkit contains the following (can be added with more things for future improvement):
            >> Number of steps
            >> Cadence
            >> Freeze Index
            >> Locomotor Freeze Index
            >> Average Step/Stride Duration
            >> Std of Step/Stride Duration
            >> Speed of Gait
            >> Symmetry
        Note:
            >> Try catch on each computation, if computation resulted in error
                data will be annotatd as null values

        Args:
            accel_dataframe (type = pd.DataFrame)

        Returns:
            RType: Dictionary 
            Returns dictionary containing features computed from PDKit Package

        """

        ## check if dataframe is valid ##
        if not isinstance(accel_dataframe, pd.DataFrame):
            raise Exception("Please parse pandas dataframe into the parameter")

        if accel_dataframe.shape[0] == 0:
            raise Exception("Dataframe is empty")

        window_start = accel_dataframe.td[0]
        window_end = accel_dataframe.td[-1]
        window_duration = window_end - window_start
        feature_dict = {}
        for axis in ["x","y","z", "AA"]:
            gp = pdkit.GaitProcessor(duration           = window_duration, 
                                     cutoff_frequency   = self.pdkit_gait_frequency_cutoff, 
                                     filter_order       = self.pdkit_gait_filter_order,
                                     delta              = self.pdkit_gait_delta, 
                                     sampling_frequency = self.sampling_frequency)
            series  = accel_dataframe[axis] 
            var     = series.var()
            try:
                if (var) < 1e-4:
                    steps   = 0
                    cadence = 0
                else:
                    strikes, _ = gp.heel_strikes(series)
                    steps      = np.size(strikes) 
                    cadence    = steps/window_duration
            except:
                steps   = 0 
                cadence = 0
            try:
                speed_of_gait = gp.speed_of_gait(series, wavelet_level = 6)   
            except:
                speed_of_gait = np.NaN
            if steps >= 2:   
                step_durations = []
                for i in range(1, np.size(strikes)):
                    step_durations.append(strikes[i] - strikes[i-1])
                avg_step_duration = np.mean(step_durations)
                sd_step_duration = np.std(step_durations)
            else:
                avg_step_duration = np.NaN
                sd_step_duration  = np.NaN

            if steps >= 4:
                strides1              = strikes[0::2]
                strides2              = strikes[1::2]
                stride_durations1     = []
                for i in range(1, np.size(strides1)):
                    stride_durations1.append(strides1[i] - strides1[i-1])
                stride_durations2     = []
                for i in range(1, np.size(strides2)):
                    stride_durations2.append(strides2[i] - strides2[i-1])
                avg_number_of_strides = np.mean([np.size(strides1), np.size(strides2)])
                avg_stride_duration   = np.mean((np.mean(stride_durations1),
                            np.mean(stride_durations2)))
                sd_stride_duration    = np.mean((np.std(stride_durations1),
                            np.std(stride_durations2)))
            else:
                avg_number_of_strides  = np.NaN
                avg_stride_duration    = np.NaN
                sd_stride_duration     = np.NaN
            try:
                step_regularity        = gp.gait_regularity_symmetry(series, average_step_duration=avg_step_duration, 
                                                            average_stride_duration=avg_stride_duration)[0]
            except:
                step_regularity        = np.NaN

            try:
                stride_regularity      = gp.gait_regularity_symmetry(series, average_step_duration=avg_step_duration, 
                                                            average_stride_duration=avg_stride_duration)[1]
            except:
                stride_regularity      = np.NaN     
            try:
                symmetry               =  gp.gait_regularity_symmetry(series, average_step_duration=avg_step_duration, 
                                                            average_stride_duration=avg_stride_duration)[2]       
            except:
                symmetry               = np.NaN
            try:
                energy_freeze_index    = calculate_freeze_index(series, 100)[0]
            except:
                energy_freeze_index    = np.NaN
            try:
                locomotor_freeze_index = calculate_freeze_index(series, 100)[1]
            except:
                locomotor_freeze_index = np.NaN

            feature_dict["%s_steps"%axis]                 = steps
            feature_dict["%s_energy_freeze_index"%axis]   = energy_freeze_index
            feature_dict["%s_loco_freeze_index"%axis]     = locomotor_freeze_index
            feature_dict["%s_avg_step_duration"%axis]     = avg_step_duration
            feature_dict["%s_sd_step_duration"%axis]      = sd_step_duration
            feature_dict["%s_walking_cadence"%axis]       = cadence
            feature_dict["%s_avg_number_of_strides"%axis] = avg_number_of_strides
            feature_dict["%s_avg_stride_duration"%axis]   = avg_stride_duration
            feature_dict["%s_sd_stride_duration"%axis]    = sd_stride_duration
            feature_dict["%s_speed_of_gait"%axis]         = speed_of_gait
            feature_dict["%s_step_regularity"%axis]       = step_regularity
            feature_dict["%s_stride_regularity"%axis]     = stride_regularity
            feature_dict["%s_symmetry"%axis]              = symmetry
        feature_dict["window_size"] = window_duration
        return feature_dict



    def run_pipeline_using_filepath(self, filepath):
        accel_data = query.get_sensor_data_from_filepath(filepath, "userAcceleration")
        rotation_data = query.get_sensor_data_from_filepath(filepath, "rotationRate")

        ## check if time series is of type dataframe                       
        if not ((isinstance(rotation_data, pd.DataFrame) and isinstance(accel_data, pd.DataFrame))):
            return "#EMPTY FILEPATH"
        if not ((rotation_data.shape[0] != 0 and (accel_data.shape[0] != 0))):
            return "#EMPTY DATAFRAME"
    
        resampled_rotation = self.gait_processor.resample_signal(rotation_data)
        resampled_accel    = self.gait_processor.resample_signal(accel_data)
        
        rotation_dict = self.get_gait_rotation_info(resampled_rotation)
        periods = [v["period"] for k,v in rotation_dict.items()]

        list_df_chunks     = self.split_gait_dataframe_to_chunks(resampled_accel, periods)
        feature_dictionary = self.featurize_gait_dataframe_chunks_by_window(list_df_chunks, rotation_dict)
        
        return feature_dictionary