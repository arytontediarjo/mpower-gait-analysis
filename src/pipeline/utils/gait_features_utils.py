## future library imports ## 
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import absolute_import

## standard library import ## 
import time
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from scipy import signal
from scipy.fftpack import (rfft, fftfreq)
from scipy.signal import (butter, lfilter, correlate, freqz)
from scipy import interpolate, signal, fft
from sklearn import metrics
from pywt import wavedec

## pdkit imports ##
import pdkit
from pdkit.utils import butter_lowpass_filter, numerical_integration

## library imports ##
import src.pipeline.utils.query_utils as query

## suppress warnings ## 
warnings.simplefilter("ignore")

class GaitFeaturize:
    """
        This is the gait featurization pipeline that is compatible for SageBionetworks synapseTable data pipeline
        The values listed as default is loosely based on compatible research, and changed based on future requirements
        and specs. 

        Args:
            rotation_frequency_cutoff (type: int, float): frequency cutoff for gyroscope data, 
                                                            2 Hz as proposed by research paper (common butterworth filter)
            
            rotation_filter_order (type: float): signal filter order with cutoff of second order (common butterworth filter)
            
            rotation_aucXt_upper_limit (type: int, float): upper limit of AUC during "zero crossing period", 
                                                            an upper limit of above
                                                            two indicates that user is doing a turn/rotation movements
            
            pdkit_gait_frequency_cutoff (type: int, float): a cutoff of frequency for low-pass filter 
                                                                on the walking data (5 Hz as indicated from their source code)
            
            pdkit_gait_filter_order (type: int, float): 4th filter order as indicated from source code
            
            pdkit_gait_delta (type: int, float): per pdkit documentation, "A point is considered a maximum peak if 
                                                    it has the maximal value, 
                                                    and was preceded (to the left) 
                                                    by a value lower by delta (0.5 default)".
            
            variance_cutoff (type: int, float): a cutoff of variance on signal data, 
                                                    if data has too low of variance, 
                                                    it will be considered as 
                                                    not moving (0 steps on window)
            window_size (type: int, float): window size of that dynamically spans 
                                                through the longitudinal data (512 sample ~ around 5 seconds)
            
            step_size (type: int, float): step size make from the previous 
                                            start point of the the window size (50 ~ around 0.5 seconds)
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
                freeze_band                 = [3,8]
                ):
        self.rotation_frequency_cutoff = rotation_frequency_cutoff
        self.rotation_filter_order = rotation_filter_order
        self.rotation_aucXt_upper_limit = rotation_aucXt_upper_limit
        self.pdkit_gait_filter_order = pdkit_gait_filter_order
        self.pdkit_gait_frequency_cutoff = pdkit_gait_frequency_cutoff
        self.pdkit_gait_delta = pdkit_gait_delta
        self.variance_cutoff = variance_cutoff
        self.window_size = window_size
        self.step_size = step_size
        self.loco_band = loco_band
        self.freeze_band = freeze_band


    def detect_zero_crossing(self, array):
        """
        Function to locate zero crossings in a time series signal data
        Args:
            array (type: np.array, list): array of number sequence (type = np.array or list)
        Returns: 
            Rtype: np.array
            Return index location before sign change in longitudinal sequence (type = N x 2 array)
            data format: [[index start, index_end], 
                        [[index start, index_end], .....]
        """
        zero_crossings = np.where(np.diff(np.sign(array)))[0]
        return zero_crossings

    def compute_rotational_features(self, accel_data, rotation_data):
        """
        Function to calculate rotational movement from gyroscope signals
        Args:
            accel_data (type: pd.DataFrame)    : columns(timeIndex(index), td (timeDifference), x, y, z, AA) accelerometer dataframe 
            rotation_data (type: pd.DataFrame) : columns(timeIndex(index), td (timeDifference), x, y, z, AA) gyroscope dataframe 
        
        Returns: 
            RType: List
            List of dictionary of gait features during rotational motion sequence
            format: [{omega: some_value, ....}, 
                    {omega: some_value, ....},....]
        """
        list_rotation = []
        for orientation in ["x", "y", "z", "AA"]:
            start = 0
            dict_list = {}
            dict_list["td"] = []
            dict_list["auc"] = []
            dict_list["turn_duration"] = []
            dict_list["aucXt"] = []
            rotation_data[orientation] = butter_lowpass_filter(data         = rotation_data[orientation], 
                                                                sample_rate = 100,  ## TODO: change this
                                                                cutoff      = self.rotation_frequency_cutoff, 
                                                                order       = self.rotation_filter_order) 
            zcr_list = self.detect_zero_crossing(rotation_data[orientation].values)
            turn_window = 0

            ## iterate through period of zero crossing list ##
            for i in zcr_list: 
                rotation_td = rotation_data["td"].iloc[start:i+1]
                rotation_rate_data = rotation_data[orientation].iloc[start:i+1]
                accel = accel_data[orientation].iloc[start:i+1]
                turn_duration = rotation_data["td"].iloc[i+1] - rotation_data["td"].iloc[start]
                
                start  = i + 1
                
                ## only take data when dataframe shape is bigger than 2 data points ##
                if (len(rotation_rate_data) >= 2): 
                    auc   = np.abs(metrics.auc(rotation_td, rotation_rate_data)) 
                    aucXt = auc * turn_duration
                    omega = auc / turn_duration

                    ## extract aucXt bigger than two ##
                    if aucXt > 2:
                        turn_window += 1

                        ## instantiate pdkit gait processor for processing gait features during rotation ##
                        gp = pdkit.GaitProcessor(duration        = turn_duration,
                                                cutoff_frequency = self.pdkit_gait_frequency_cutoff,
                                                filter_order     = self.rotation_filter_order,
                                                delta            = self.pdkit_gait_delta)
                        
                        ## try-except each pdkit features, if error fill with zero
                        ## TODO: most of the features are susceptible towards error e.g. not enough heel strikes
                        ## should it be remove instead???????
                        try: 
                            strikes, _ = gp.heel_strikes(accel)
                            steps      = np.size(strikes)
                            cadence    = steps/turn_duration
                        except:
                            steps = 0
                            cadence = 0
                        try:
                            peaks_data = accel.values
                            maxtab, _ = peakdet(peaks_data, gp.delta)
                            x = np.mean(peaks_data[maxtab[1:,0].astype(int)] - peaks_data[maxtab[:-1,0].astype(int)])
                            frequency_of_peaks = abs(1/x)
                        except:
                            frequency_of_peaks = 0

                        ## condition if steps are more than 2, during 2.5 seconds window  ##
                        if steps >= 2:   
                            step_durations = []
                            for i in range(1, np.size(strikes)):
                                step_durations.append(strikes[i] - strikes[i-1])
                            avg_step_duration = np.mean(step_durations)
                            sd_step_duration = np.std(step_durations)
                        else:
                            avg_step_duration = 0
                            sd_step_duration = 0

                        ## condition if steps are more than 4, can get stride features ##
                        if steps >= 4:
                            strides1 = strikes[0::2]
                            strides2 = strikes[1::2]
                            stride_durations1 = []
                            for i in range(1, np.size(strides1)):
                                stride_durations1.append(strides1[i] - strides1[i-1])
                            stride_durations2 = []
                            for i in range(1, np.size(strides2)):
                                stride_durations2.append(strides2[i] - strides2[i-1])
                            avg_number_of_strides = np.mean([np.size(strides1), np.size(strides2)])
                            avg_stride_duration = np.mean((np.mean(stride_durations1),
                                        np.mean(stride_durations2)))
                            sd_stride_duration = np.mean((np.std(stride_durations1),
                                        np.std(stride_durations2)))
                        else:
                            avg_number_of_strides = 0
                            avg_stride_duration = 0
                            sd_stride_duration = 0
                        
                        list_rotation.append({
                                "rotation.axis"                 : orientation,
                                "rotation.energy_freeze_index"  : self.calculate_freeze_index(accel)[0],
                                "rotation.window_duration"      : turn_duration,
                                "rotation.auc"                  : auc,      ## radian
                                "rotation.omega"                : omega,    ## radian/secs 
                                "rotation.aucXt"                : aucXt,    ## radian . secs (based on research paper)
                                "rotation.window_start"         : rotation_td[0],
                                "rotation.window_end"           : rotation_td[-1],
                                "rotation.num_window"           : turn_window,
                                "rotation.avg_step_duration"    : avg_step_duration,
                                "rotation.sd_step_duration"     : sd_step_duration,
                                "rotation.steps"                : steps,
                                "rotation.cadence"              : cadence,
                                "rotation.frequency_of_peaks"   : frequency_of_peaks,
                                "rotation.avg_number_of_strides": avg_number_of_strides,
                                "rotation.avg_stride_duration"  : avg_stride_duration,
                                "rotation.sd_stride_duration"   : sd_stride_duration
                        })
        return list_rotation


    def split_dataframe_to_dict_chunk_by_interval(self, accel_data, rotation_data):
        """
        A function to separate dataframe to several chunks separated by rotational motion
        done by a subject. 
        Parameter:
            accel_data    (pd.DataFrame): formatted-columns accelerometer dataframe 
            rotation_data (pd.DataFrame): formatted-columns rotation dataframe
        
        Returns: 
            RType: Dictionary (contains chunk of dataframes)
            A dictionary mapping of data chunks of non-rotational motion
            Return format: {"chunk1": pd.DataFrame, 
                            "chunk2": pd.DataFrame, etc ......}
        """
        if (not isinstance(accel_data, pd.DataFrame)):
            raise Exception("please use dataframe for acceleration")
        data_chunk = {}
        window = 1 
        last_stop = 0
        #if no rotation#
        if len(rotation_data) == 0 :
            data_chunk["chunk1"] = accel_data
            return data_chunk
        rotation_data = pd.DataFrame(rotation_data)
        for start, end in rotation_data[["rotation.window_start", "rotation.window_end"]].values:
            if start <= 0:
                last_stop = end
                continue
            ## edge case -> if rotation is overlapping with start, jump to end of rotation ##
            if last_stop >= start:
                last_stop = end
                continue
            ## ideal case ## 
            data_chunk["chunk%s"%window] = accel_data[(accel_data["td"]<=start) & (accel_data["td"]>=last_stop)]
            last_stop = end
            window += 1
        ## edge case -> after the last rotation, take the rest ## 
        if last_stop < accel_data["td"][-1]:
            data_chunk["chunk%s"%str(window)] = accel_data[(accel_data["td"]>=end)]
        return data_chunk

    def compute_pdkit_feature_per_window(self, data):
        """
        A modified function to calculate feature per smaller time window chunks
        Args: 
            data (pd.DataFrame) : dataframe of longitudinal signal data
            orientation (str)   : coordinate orientation of the time series
        Returns:
            RType: List
            returns list of dict of walking features using PDKIT package
            Return format: [{steps: some_value, ...}, 
                            {steps: some_value, ...}, ...]
        """
        ts_arr      = []
        for orientation in ["x", "y", "z", "AA"]:
            ts = data.copy()
            window_size = self.window_size
            step_size   = self.step_size
            jPos        = window_size + 1
            i           = 0
            if len(ts) < jPos:
                ts_arr.append(self.generate_pdkit_features_in_dict(ts, orientation))
                continue
            while jPos < len(ts):
                jStart = jPos - window_size
                subset = ts.iloc[jStart:jPos]
                ts_arr.append(self.generate_pdkit_features_in_dict(subset, orientation)) 
                jPos += step_size
                i = i + 1
        return ts_arr

    def generate_pdkit_features_in_dict(self, data, orientation):
        """
        Function to generate pdkit features given orientation and time-series dataframe
        Args:
            `data` (pd.DataFrame): dataframe of time series (timeIndex, x, y, z, AA, td)
            `orientation`   (str): axis oriention of walking (type: str)
        Returns:
            RType: Dictionary
            Returns a dictionary mapping of pdkit features in walking/balance data
        """
        window_duration = data.td[-1] - data.td[0]
        accel = data[orientation]
        var = accel.var()
        gp = pdkit.GaitProcessor(duration        = window_duration,
                                cutoff_frequency = self.pdkit_gait_frequency_cutoff,
                                filter_order     = self.pdkit_gait_filter_order,
                                delta            = self.pdkit_gait_delta)
        try:
            if (var) < self.variance_cutoff:
                steps = 0
                cadence = 0
            else:
                strikes, _ = gp.heel_strikes(accel)
                steps      = np.size(strikes) 
                cadence    = steps/window_duration
        except:
            steps = 0  
            cadence = 0
        try:
            peaks_data = accel.values
            maxtab, _ = peakdet(peaks_data, gp.delta)
            x = np.mean(peaks_data[maxtab[1:,0].astype(int)] - peaks_data[maxtab[:-1,0].astype(int)])
            frequency_of_peaks = abs(1/x)
            if np.isnan(frequency_of_peaks):
                frequency_of_peaks = 0
        except:
            frequency_of_peaks = 0
        try:
            speed_of_gait = gp.speed_of_gait(accel)   
        except:
            speed_of_gait = 0
        if steps >= 2:   # condition if steps are more than 2, during 2.5 seconds window 
            step_durations = []
            for i in range(1, np.size(strikes)):
                step_durations.append(strikes[i] - strikes[i-1])
            avg_step_duration = np.mean(step_durations)
            sd_step_duration = np.std(step_durations)
        else:
            avg_step_duration = 0
            sd_step_duration = 0

        if steps >= 4:
            strides1 = strikes[0::2]
            strides2 = strikes[1::2]
            stride_durations1 = []
            for i in range(1, np.size(strides1)):
                stride_durations1.append(strides1[i] - strides1[i-1])
            stride_durations2 = []
            for i in range(1, np.size(strides2)):
                stride_durations2.append(strides2[i] - strides2[i-1])
            avg_number_of_strides = np.mean([np.size(strides1), np.size(strides2)])
            avg_stride_duration = np.mean((np.mean(stride_durations1),
                        np.mean(stride_durations2)))
            sd_stride_duration = np.mean((np.std(stride_durations1),
                        np.std(stride_durations2)))
        else:
            avg_number_of_strides = 0
            avg_stride_duration = 0
            sd_stride_duration = 0

        try:
            step_regularity   = gp.gait_regularity_symmetry(accel,average_step_duration=avg_step_duration, 
                                                        average_stride_duration=avg_stride_duration)[0]
        except:
            step_regularity   = 0
        
        try:
            stride_regularity = gp.gait_regularity_symmetry(accel,average_step_duration=avg_step_duration, 
                                                        average_stride_duration=avg_stride_duration)[1]
        except:
            stride_regularity = 0     

        try:
            symmetry          =  gp.gait_regularity_symmetry(accel,average_step_duration=avg_step_duration, 
                                                        average_stride_duration=avg_stride_duration)[2]       
        except:
            symmetry          = 0                                                                                                             
        
        feature_dict = {
                "walking.window_duration"      : window_duration,
                "walking.window_start"
                "walking.axis"                 : orientation,
                "walking.energy_freeze_index"  : self.calculate_freeze_index(accel)[0],
                "walking.avg_step_duration"    : avg_step_duration,
                "walking.sd_step_duration"     : sd_step_duration,
                "walking.steps"                : steps,
                "walking.cadence"              : cadence,
                "walking.frequency_of_peaks"   : frequency_of_peaks,
                "walking.avg_number_of_strides": avg_number_of_strides,
                "walking.avg_stride_duration"  : avg_stride_duration,
                "walking.sd_stride_duration"   : sd_stride_duration,
                "walking.speed_of_gait"        : speed_of_gait,
                "walking.step_regularity"      : step_regularity,
                "walking.stride_regularity"    : stride_regularity,
                "walking.symmetry"             : symmetry}
        return feature_dict

    def calculate_freeze_index(self, series):
        """
        Modified pdkit FoG function; 
            *removed resampling from the signal
        Args: 
            `series` (pd.Series): pd.Series of signal in one orientation
        Returns:
            array of [freeze index , sumLocoFreeze]
        """
        loco_band   = self.loco_band
        freeze_band = self.freeze_band
        window_size = series.shape[0]
        sampling_frequency = 100
        f_res = sampling_frequency / window_size
        f_nr_LBs = int(loco_band[0] / f_res)
        f_nr_LBe = int(loco_band[1] / f_res)
        f_nr_FBs = int(freeze_band[0] / f_res)
        f_nr_FBe = int(freeze_band[1] / f_res)

        ## normalize series  ## 
        series = series - np.mean(series)

        ## discrete fast fourier transform ##
        Y = np.fft.fft(series, int(window_size))
        Pyy = abs(Y*Y) / window_size
        areaLocoBand = numerical_integration( Pyy[f_nr_LBs-1 : f_nr_LBe], sampling_frequency)
        areaFreezeBand = numerical_integration( Pyy[f_nr_FBs-1 : f_nr_FBe], sampling_frequency)
        sumLocoFreeze = areaFreezeBand + areaLocoBand
        freezeIndex = areaFreezeBand / areaLocoBand
        return freezeIndex, sumLocoFreeze
        



    def walk_feature_pipeline(self, filepath):
        """
        Function for walk feature pipeline, which consist of features from pdkit package 
        that has been generated as a list. 
        Pipeline cleaning process:
            >> Zero steps on very low variance signal (10e-3)
            >> Segmenting data from rotational movement
        Args:
            `filepath`    (str) : string of filepath to /.synapseCache 
        Returns:
            RType: List
            return combined list of walking data from several chunks
        """    
        accel_ts    = query.get_sensor_ts_from_filepath(filepath = filepath, 
                                            sensor = "userAcceleration")
        rotation_ts = query.get_sensor_ts_from_filepath(filepath = filepath, 
                                            sensor = "rotationRate")
        # return errors # 
        if not ((isinstance(rotation_ts, pd.DataFrame) and isinstance(accel_ts, pd.DataFrame))):
            return "#ERROR"
        rotation_occurence = self.compute_rotational_features(accel_ts, rotation_ts)
        gait_dict = self.split_dataframe_to_dict_chunk_by_interval(accel_ts, rotation_occurence)
        gait_feature_arr = []
        for chunks in gait_dict.keys():
            gait_feature_arr.append(self.compute_pdkit_feature_per_window(data = gait_dict[chunks]))
        gait_feature_arr = [j for i in gait_feature_arr for j in i]
        if len(gait_feature_arr) == 0:
            return "#ERROR"
        return gait_feature_arr


    def rotation_feature_pipeline(self, filepath):
        """
        Function for featurizing rotation pipeline, which consist of features from pdkit package 
        that has been generated as a list on the rotation sequence 
        Pipeline cleaning process:
            >> Zero steps on very low variance signal (10e-3)
            >> Segmenting data from rotational movement
        Args:
            `filepath`    (str) : string of filepath to .synapseCache 
        Returns:
            RType: List
            return combined list of walking data from several chunks
        """  
        rotation_ts = query.get_sensor_ts_from_filepath(filepath = filepath, 
                                                        sensor = "rotationRate")
        accel_ts = query.get_sensor_ts_from_filepath(filepath = filepath, 
                                                        sensor = "userAcceleration")                                  
        if not ((isinstance(rotation_ts, pd.DataFrame) and isinstance(accel_ts, pd.DataFrame))):
            return "#ERROR"
        rotation_ts = self.compute_rotational_features(accel_ts, rotation_ts)
        if len(rotation_ts) == 0:
            return "#ERROR"
        return rotation_ts

    def annotate_consecutive_zeros(self, data, feature):
        """
        Function to annotate consecutive zeros in a dataframe

        Args:
            `data`    : dataframe
            `feature` : feature to assess on counting consecutive zeros
        
        returns:
            A new column-series of data with counted consecutive zeros (if available)
        """
        step_shift_measure = data[feature].ne(data[feature].shift()).cumsum()
        counts = data.groupby(['recordId', step_shift_measure])[feature].transform('size')
        data['consec_zero_steps_count'] = np.where(data[feature].eq(0), counts, 0)
        return data

    def featurize_wrapper(self, data):
        """
        wrapper function for multiprocessing jobs (walking/balance)
        Args:
            `data` (pd.DataFrame): takes in pd.DataFrame
        returns a json file featurized walking data
        """
        data["gait.walk_features"] = data["gait.json_pathfile"].apply(self.walk_feature_pipeline)
        data["gait.rotation_features"] = data["gait.json_pathfile"].apply(self.rotation_feature_pipeline)
        return data


