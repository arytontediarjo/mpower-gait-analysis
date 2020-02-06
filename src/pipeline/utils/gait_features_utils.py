"""
Featurization Pipeline Class on Gait Signal Data
"""

# future liibrary imports ## 
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

class GaitFeaturize:

    """
    This is the gait featurization package that combines features from pdkit package,
    rotation detection algorithm from gyroscope, and gait pipeline with overlapped-moving window.
    
    References:
        Rotation-Detection Paper : https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5811655/
        PDKit Docs               : https://pdkit.readthedocs.io/_/downloads/en/latest/pdf/
        PDKit Gait Source Codes  : https://github.com/pdkit/pdkit/blob/79d6127454f22f7ea352a2540c5b8364b21356e9/pdkit/gait_processor.py
        Freeze of Gait Docs      : https://ieeexplore.ieee.org/document/5325884

    Args:
        gyro_frequency_cutoff (dtype: int, float) : frequency cutoff for gyroscope rotation rate
        gyro_filter_order (dtype: float)          : signal filter order for gyroscope rotation rate
        gyro_aucXt_upper_limit (dtype: int, float): upper limit of AUC during "zero crossing period", 
                                                    an upper limit of above two indicates that user is 
                                                    doing a turn/rotation movements (per Rotation Detection Paper)
        accel_frequency_cutoff (dtype: int, float): Frequency cutoff on user acceleration (default = 5Hz)
        accel_filter_order (dtype: int, float)    : Signal filter order on user acceleration (default = 4th Order)
        accel_delta (dtype: int, float)           : A point is considered a maximum peak if it has the maximal value, 
                                                    and was preceded (to the left) by a value lower by delta (0.5 default).
        variance_cutoff (dtype: int, float)       : a cutoff of variance on signal data, 
                                                    if data has too low of variance, 
                                                    it will be considered as 
                                                    not moving (0 steps on window)
        window_size (dtype: int, float)           : window size of that dynamically spans 
                                                    through the longitudinal data (default: 512 sample ~ around 5 seconds)
        step_size (dtype: int, float)             : step size make from the previous 
                                                    start point of the the window size (50 ~ around 0.5 seconds)
        loco_band (dtype: list)                   : The ratio of the energy in the locomotion band, measured in Hz ([0.5, 3] default)
        freeze_band (dtype: list)                 : The ration of energy in the freeze band, measured in Hz ([3, 8] default)
        sampling_frequency (dtype: float, int)    : Samples collected per seconds

    HOW-TO-USE:
        import utils.gait_features_utils
        sensor_data =  (options of pd.DataFrame or string filepath) 
        gf          =  gait_features_utils.GaitFeaturize()
        features    =  gf.run_gait_feature_pipeline(sensor_data)

    """
    
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
        


    def get_sensor_data(self, data, sensor): 
        """
        Utility Function to get sensor data given a filepath.
        Data extraction will be adjusted to several different
        json formatting of its key-value pairs based on mPower Synapse Table        
        
        Args: 
            data     (dtype: string, pd.DataFrame): string of filepath or pandas DataFrame
            sensor   (dtype: string)              : the sensor type (userAcceleration, 
                                                    acceleration with gravity, 
                                                    gyroscope etc. from time series)
        Returns:
            Rtype: pd.DataFrame
            Return a formatted version of the dataframe that contains an index of time-index dataframe (timestamp), 
            and columns of time differences in seconds, and sensor measurement in x, y, z coordinate from the filepaths.
            Errors will be annotated with its reason, and instantiated with "#ERROR:...."
        """
        ## if empty filepaths return it back ##
        
        if isinstance(data, str):
            if data == "#ERROR: EMPTY FILEPATH":
                return data
        
            ## open filepath ##
            try:
                with open(data) as f:
                    json_data = f.read()
                    data = pd.DataFrame(json.loads(json_data))
            except:
                return "#ERROR: FILE CANNOT BE PARSED TO DATAFRAME"

        ## return accelerometer data back if empty ##
        if data.shape[0] == 0: 
            return "#ERROR: EMPTY DATAFRAME"
        
        ## retrieve data if sensor is stored as a long format ##
        if ("sensorType" in data.columns):
            try:
                data = data[data["sensorType"] == sensor]
            except:
                return "#ERROR: SENSOR NOT AVAILABLE"
        
        ## retrieve data if sensor is stored as a wide format ##
        else:
            try:
                data = data[["timestamp", sensor]]
                data["x"] = data[sensor].apply(lambda x: x["x"])
                data["y"] = data[sensor].apply(lambda x: x["y"])
                data["z"] = data[sensor].apply(lambda x: x["z"])
                data = data.drop([sensor], axis = 1)
            except:
                return "#ERROR: SENSOR NOT AVAILABLE"

        ## if the shape of data is zero after transformation, return sensor not available ##
        if data.shape[0] == 0:
            return "#ERROR: SENSOR NOT AVAILABLE"
        
        ## format dataframe to desired pdkit format ##
        return self.format_time_series_data(data)

        

    def format_time_series_data(self, data):
        """
        Utility function to clean accelerometer data to PDKIT format
        Format => [timestamp (DatetimeIndex), x, y, z , AA, td]

        time (dtype: DatetimeIndex) : An index of time in seconds
        x  (dtype: float64) : sensor-values in x axis
        y  (dtype: float64) : sensor-values in y axis
        z  (dtype: float64) : sensor-values in z axis
        AA (dtype: float64) : resultant of sensor values
        td (dtype: float64) : current time difference from zero in seconds

        Args: 
            data(type: pd.DataFrame): pandas dataframe of time series
        
        Returns:
            RType: pd.DataFrame
            Returns a formatted dataframe 
        """
        if data.shape[0] == 0:
            raise Exception("#ERROR: EMPTY DATAFRAME")
        data = data.dropna(subset = ["x", "y", "z"])
        date_series = pd.to_datetime(data["timestamp"], unit = "s")
        data["td"] = (date_series - date_series.iloc[0]).apply(lambda x: x.total_seconds())
        data["timestamp"] = data["td"]
        data = data.set_index("timestamp")
        data.index = pd.to_datetime(data.index, unit = "s")
        data["AA"] = np.sqrt(data["x"]**2 + data["y"]**2 + data["z"]**2)

        ## some sanity checking ##
        
        ## remove data duplicates ##
        data = data[~data.index.duplicated(keep='first')]

        ## sort all indexes ## 
        data = data.sort_index()
        return data[["td","x","y","z","AA"]] 


    def calculate_freeze_index(self, accel_series, sample_rate):
        """
        Modified pdkit FoG freeze index function to be compatible with 
        current source dode
        Args: 
            accel_series (dtype = pd.Series): pd.Series of acceleration signal in one axis
            sample_rate  (dtype = float64)  : signal sampling rate
        Returns:
            RType: List
            List containing 2 values of freeze index values [energy of freeze index, locomotor freeze index]
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
        Function to output rotation information (omega, period of rotation) 
        in a dictionary format, given a gyroscope dataframe and its axis orientation

        Args:
            gyro_dataframe (type: pd.DataFrame)       : A gyrosocope dataframe
            axis           (type: str, default = "y") : Axis of desired rotation orientation,
                                                        default is y axis (pocket-test), 
                                                        unless specified otherwise                                  
        Returns:
            RType: Dictionary
            Return format {"rotation_chunk_1" : {omega: some_value, period: some_value},
                            "rotation_chunk_2": {omega: some_value, period: some_value}}
        """

        ## check if dataframe is valid ## 
        if not isinstance(gyro_dataframe, pd.DataFrame):
            raise Exception("PIPELINE ERROR: PLEASE PARSE PANDAS DATAFRAME")

        if gyro_dataframe.shape[0] == 0:
            raise Exception("PIPELINE ERROR: DATAFRAME IS EMPTY")

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
        Function to chunk dataframe into several periods of rotational and non-rotational sequences into List.
        Each dataframe chunk in the list will be in dictionary format containing information of the dataframe itself
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
            raise Exception("PIPELINE ERROR: PLEASE PARSE PANDAS DATAFRAME")

        if accel_dataframe.shape[0] == 0:
            raise Exception("PIPELINE ERROR: DATAFRAME IS EMPTY")

        ## instantiate initial values ## 
        chunk_list          =  []
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
        Function to featurize list of rotation-segmented dataframe chunks with moving windows,
        and returns list of dictionary with all the pdkit and rotation features from each moving window
        Notes:
          >> if dataframe chunk is smaller than the window size or is a rotation dataframe chunk
              it will just be treated as one window
          >> if dataframe is bigger than designated window, then a while loop of gait featurization
             using moving window will occur
    
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
                feature_dict = self.get_pdkit_gait_features(curr_dataframe)
                feature_dict["rotation_omega"] = rotation_omega
                feature_dict["window"]         = "window_%s"%num_window
                feature_list.append(feature_dict)
                num_window += 1
            else:
                while jPos < len(curr_dataframe):
                    jStart     = jPos - window_size
                    subset     = curr_dataframe.iloc[jStart:jPos]
                    feature_dict = self.get_pdkit_gait_features(subset)
                    feature_dict["rotation_omega"] = rotation_omega
                    feature_dict["window"]         = "window_%s"%num_window
                    feature_list.append(feature_dict)
                    jPos += step_size
                    num_window += 1   
        return feature_list


    def get_pdkit_gait_features(self, accel_dataframe):
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
            raise Exception("PIPELINE ERROR: PLEASE PARSE PANDAS DATAFRAME INTO PARAMETER")

        if accel_dataframe.shape[0] == 0:
            raise Exception("PIPELINE ERROR: DATAFRAME IS EMPTY")

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

    
    def resample_signal(self, dataframe):
        """
        Utility method for data resampling,
        Data will be interpolated using linear method

        Args: 
            dataframe: A time-indexed dataframe
        
        Returns:
            RType: pd.DataFrame
            Returns a resampled dataframe based on predefined sampling frequency on class instantiation
        """
        new_freq = np.round(1 / self.sampling_frequency, decimals=6)
        df_resampled = dataframe.resample(str(new_freq) + 'S').mean()
        df_resampled = df_resampled.interpolate(method='linear') 
        return df_resampled


    def run_gait_feature_pipeline(self, data):
        """
        main entry point of this featurizaton class, parameter will take in pd.DataFrame or the filepath to the dataframe.

        Args:
            data (dtype: dataframe or string filepath): contains dataframe (or filepath to the dataframe) 
                                                        that contains at least columns of [timestamp, x, y, z]
                                                        of gyroscope and user acceleration data
        Returns:
            A list of dictionary. With each dictionary representing gait features on one window.
        """

        accel_data    = self.get_sensor_data(data, "userAcceleration")
        rotation_data = self.get_sensor_data(data, "rotationRate")        
        
        ## Check if time series is of type dataframe and not an empty dataframe                     
        if not ((isinstance(rotation_data, pd.DataFrame) and isinstance(accel_data, pd.DataFrame))):
            return "#ERROR: EMPTY FILEPATH"
        if not ((rotation_data.shape[0] != 0 and (accel_data.shape[0] != 0))):
            return "#ERROR: EMPTY DATAFRAME"

        resampled_rotation         = self.resample_signal(rotation_data)
        resampled_accel            = self.resample_signal(accel_data)
        rotation_dict              = self.get_gait_rotation_info(resampled_rotation)
        periods                    = [v["period"] for k,v in rotation_dict.items()]
        list_df_chunks             = self.split_gait_dataframe_to_chunks(resampled_accel, periods)
        list_of_feature_dictionary = self.featurize_gait_dataframe_chunks_by_window(list_df_chunks, rotation_dict)
        return list_of_feature_dictionary