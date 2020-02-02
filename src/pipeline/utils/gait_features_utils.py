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
    """
        This is the gait featurization package that combines features from pdkit package,
        rotation detection algorithm from gyroscope, and gait pipeline with overlapped-moving window.

        - Feature pipeline in summary - 
        
        >>  Takes in pdkit formatted time-series gyroscope and userAcceleration
            pd.dataFrame (pd.DatetimeIndex (unit: s), x, y, z, AA, td (time difference from start (unit: s))) 
        >>  Detect turning motion from gyroscope data by assessing rotation rate 
            AUC and duration during zero-crossing period. 
            If rotation rate AUC * rotation rate duration > <some particular value (2 based on paper)>,
            zero crossing period will be considered a turning motion
        >>  Extract rotational motion features and its gait features using pdkit package during rotation period
        >>  Remove rotational/turning motion period from walking time-series data, 
            and separate signal data into several non-rotational walking/balance data
        >>  For each chunk, iterate each chunk using a smaller window <defined by user> 
            that moves with step size <defined by user> from the starting point;
            the moving window will compute gait features iteratively, until each part of the chunk has been computed

        Note: 
                For each recordIds, features will be returned as a list of dictionary, 
                each of the dictionary inside the list represents one window of feature computation.
                Afterwards The list of dictionary for each recordIds will be normalized
                to a more consistent column features. 
                 
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
                freeze_band                 = [3,8],
                sampling_frequency          = 100
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
        self.sampling_frequency = sampling_frequency
        self.gait_processor = pdkit.processor.Processor(sampling_frequency = self.sampling_frequency)

    
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
        Function to detect rotational movement from gyroscope rotation rate (rad/s^2)
        and calculate gait features during rotational periods.

        Args:
            accel_data (type: pd.DataFrame)    : formatted columns(timeIndex(index) in seconds, 
                                                                    td (timeDifference from start) seconds, 
                                                                    x, y, z, AA) userAcceleration dataframe 
            rotation_data (type: pd.DataFrame) : formatted columns(timeIndex(index) in seconds, 
                                                                    td (timeDifference from start) in seconds, 
                                                                    x, y, z, AA) rotationRate dataframe 
        
        Returns: 
            RType: List
            List of dictionary of gait features during rotational motion sequence
            format: [{feature1: some_value, feature2: some_value}, 
                    {feature1: some_value, feature2: some_value},....]
        """

        ## check if data is formatted to requirements
        list_rotation = []
        
        for axis in ["x", "y", "z", "AA"]:
            start = 0
            dict_list = {}
            dict_list["td"] = []
            dict_list["auc"] = []
            dict_list["window_duration"] = []
            dict_list["aucXt"] = []
            rotation_data[axis] = butter_lowpass_filter(data        = rotation_data[axis], 
                                                        sample_rate = self.sampling_frequency,
                                                        cutoff      = self.rotation_frequency_cutoff, 
                                                        order       = self.rotation_filter_order) 
            zcr_list = self.detect_zero_crossing(rotation_data[axis].values)
            num_window = 0

            ## iterate through list of zero crossing period ##
            for i in zcr_list: 
                rotation_period_data    = rotation_data[[axis, "td"]].iloc[start:i+1]
                accel_subset            = accel_data[axis].iloc[start:i+1]
                window_start            = rotation_data["td"].iloc[start]  
                window_end              = rotation_data["td"].iloc[i+1]
                window_duration         = window_end - window_start   
                start  = i + 1
                
                ## only take data when dataframe shape is bigger than 2 data points ##
                if (len(rotation_period_data[axis]) >= 2): 
                    auc   = np.abs(metrics.auc(rotation_period_data["td"], rotation_period_data[axis])) 
                    aucXt = auc * window_duration
                    omega = auc / window_duration

                    ## extract aucXt bigger than two ##
                    if aucXt > 2:
                        num_window += 1

                        ## instantiate pdkit gait processor for processing gait features during rotation ##
                        gp = pdkit.GaitProcessor(duration          = window_duration,
                                                cutoff_frequency   = self.pdkit_gait_frequency_cutoff,
                                                filter_order       = self.rotation_filter_order,
                                                delta              = self.pdkit_gait_delta,
                                                sampling_frequency = self.sampling_frequency)
                        
                        ## try-except each pdkit features, if error fill with zero
                        ## TODO: most of the features are susceptible towards error e.g. not enough heel strikes
                        
                        ## resample isolated user acceleration data to 100 Hz ##
                        try: 
                            strikes, _ = gp.heel_strikes(accel_subset)
                            steps      = np.size(strikes)
                            cadence    = steps/window_duration
                        except:
                            steps = 0
                            cadence = 0
                        try:
                            peaks_data = accel_subset.values
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
                                "rotation_axis"                 : axis,
                                "rotation_energy_freeze_index"  : self.calculate_freeze_index(accel_subset, self.sampling_frequency)[0],
                                "rotation_window_duration"      : window_duration,
                                "rotation_window_start"         : window_start,
                                "rotation_window_end"           : window_end,
                                "rotation_auc"                  : auc,      ## radian
                                "rotation_omega"                : omega,    ## radian/secs 
                                "rotation_aucXt"                : aucXt,    ## radian . secs (based on research paper)
                                "rotation_num_window"           : num_window,
                                "rotation_avg_step_duration"    : avg_step_duration,
                                "rotation_sd_step_duration"     : sd_step_duration,
                                "rotation_steps"                : steps,
                                "rotation_cadence"              : cadence,
                                "rotation_frequency_of_peaks"   : frequency_of_peaks,
                                "rotation_avg_number_of_strides": avg_number_of_strides,
                                "rotation_avg_stride_duration"  : avg_stride_duration,
                                "rotation_sd_stride_duration"   : sd_stride_duration
                        })
        return list_rotation


    def split_dataframe_to_dict_chunk_by_interval(self, accel_data, rotation_data):
        """
        A function to separate dataframe into several chunks by interval of rotational motion 
        from rotation motion.
        
        Args:
            accel_data (type: pd.DataFrame)    : formatted columns(timeIndex(index) in miliseconds, 
                                                                    td (timeDifference) in miliseconds, 
                                                                    x, y, z, AA) userAcceleration dataframe 
            rotation_data (type: pd.DataFrame) : formatted columns(timeIndex(index) in miliseconds, 
                                                                    td (timeDifference) in miliseconds, 
                                                                    x, y, z, AA) rotationRate dataframe 
        
        Returns: 
            RType: Dictionary (contains chunk of dataframes)
            A dictionary mapping of data chunks of non-rotational motion
            Return format (given one rotation motion): {"chunk1": pd.DataFrame, 
                                                        "chunk2": pd.DataFrame}
        """
        
        data_chunk = {}
        window = 1 
        last_stop = 0
        #if no rotation#
        if len(rotation_data) == 0 :
            data_chunk["chunk1"] = accel_data
            return data_chunk
        rotation_data = pd.DataFrame(rotation_data)
        for start, end in rotation_data[["rotation_window_start", "rotation_window_end"]].values:
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
            Return format: [{feature1: some_value, ...}, 
                            {feature1: some_value, ...}, ...]
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
            data (pd.DataFrame): formatted dataframe of time series (timeIndex, x, y, z, AA, td)
            orientation   (str): axis oriention of walking (type: str)
        
        Returns:
            RType: Dictionary
            Returns a dictionary mapping of pdkit features in walking/balance data
        """
        window_start = data.td[0]
        window_end = data.td[-1]
        window_duration = window_end - window_start
        data = data[orientation]
        var = data.var()
        gp = pdkit.GaitProcessor(duration          = window_duration,
                                cutoff_frequency   = self.pdkit_gait_frequency_cutoff,
                                filter_order       = self.pdkit_gait_filter_order,
                                delta              = self.pdkit_gait_delta,
                                sampling_frequency = self.sampling_frequency)
        try:
            if (var) < self.variance_cutoff:
                steps = 0
                cadence = 0
            else:
                strikes, _ = gp.heel_strikes(data)
                steps      = np.size(strikes) 
                cadence    = steps/window_duration
        except:
            steps = 0  
            cadence = 0
        try:
            peaks_data = data.values
            maxtab, _ = peakdet(peaks_data, gp.delta)
            x = np.mean(peaks_data[maxtab[1:,0].astype(int)] - peaks_data[maxtab[:-1,0].astype(int)])
            frequency_of_peaks = abs(1/x)
            if np.isnan(frequency_of_peaks):
                frequency_of_peaks = 0
        except:
            frequency_of_peaks = 0
        try:
            speed_of_gait = gp.speed_of_gait(data)   
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
            step_regularity   = gp.gait_regularity_symmetry(data, average_step_duration=avg_step_duration, 
                                                        average_stride_duration=avg_stride_duration)[0]
        except:
            step_regularity   = 0
        
        try:
            stride_regularity = gp.gait_regularity_symmetry(data, average_step_duration=avg_step_duration, 
                                                        average_stride_duration=avg_stride_duration)[1]
        except:
            stride_regularity = 0     

        try:
            symmetry          =  gp.gait_regularity_symmetry(data, average_step_duration=avg_step_duration, 
                                                        average_stride_duration=avg_stride_duration)[2]       
        except:
            symmetry          = 0                                                                                                             
        
        feature_dict = {
                "walking_window_duration"      : window_duration,
                "walking_window_start"         : window_start,
                "walking_window_end"           : window_end,
                "walking_axis"                 : orientation,
                "walking_energy_freeze_index"  : self.calculate_freeze_index(data, self.sampling_frequency)[0],
                "walking_avg_step_duration"    : avg_step_duration,
                "walking_sd_step_duration"     : sd_step_duration,
                "walking_steps"                : steps,
                "walking_cadence"              : cadence,
                "walking_frequency_of_peaks"   : frequency_of_peaks,
                "walking_avg_number_of_strides": avg_number_of_strides,
                "walking_avg_stride_duration"  : avg_stride_duration,
                "walking_sd_stride_duration"   : sd_stride_duration,
                "walking_speed_of_gait"        : speed_of_gait,
                "walking_step_regularity"      : step_regularity,
                "walking_stride_regularity"    : stride_regularity,
                "walking_symmetry"             : symmetry}
        return feature_dict

    def calculate_freeze_index(self, series, accel_sample_rate):
        """
        Modified pdkit FoG freeze index function to be compatible with 
        current source code.  
        
        Args: 
            series          (type = pd.Series): pd.Series of signal in one orientation
            accel_sample_rate (type = float64): acceleration sampling rate
        
        Returns:
            array of [freeze index , sumLocoFreeze]
        """
        loco_band   = self.loco_band
        freeze_band = self.freeze_band
        window_size = series.shape[0]
        f_res       = accel_sample_rate / window_size
        f_nr_LBs    = int(loco_band[0] / f_res)
        f_nr_LBe    = int(loco_band[1] / f_res)
        f_nr_FBs    = int(freeze_band[0] / f_res)
        f_nr_FBe    = int(freeze_band[1] / f_res)

        ## normalize series  ## 
        series = series - np.mean(series)

        ## discrete fast fourier transform ##
        Y = np.fft.fft(series, int(window_size))
        Pyy = abs(Y*Y) / window_size
        areaLocoBand = numerical_integration( Pyy[f_nr_LBs-1 : f_nr_LBe], accel_sample_rate)
        areaFreezeBand = numerical_integration( Pyy[f_nr_FBs-1 : f_nr_FBe], accel_sample_rate)

        ## edge case: if integration of locomotor band is too small or zero, return #ERROR## 
        ## freeze index during this time window will not be assessed
        if areaLocoBand == 0:
            freezeIndex   = "#ERROR"
            sumLocoFreeze = "#ERROR"
            return freezeIndex, sumLocoFreeze 
        
        ## ideal case ##  
        sumLocoFreeze = areaFreezeBand + areaLocoBand
        freezeIndex = areaFreezeBand / areaLocoBand
        return freezeIndex, sumLocoFreeze

    def gait_feature_pipeline(self, filepath):
        """
        Function for gait feature pipeline, which consists gait features from pdkit package.
        Pipeline cleaning process:
            >> Zero steps on very low variance signal (10e-3)
            >> Segmenting data from rotational movement
            >> Featurize segmented data
            >> Segmented data will be featurized using pdkit
        
        Args:
            filepath    (str) : string of filepath to .synapseCache 
        
        Returns:
            RType: List
            Return combined list of walking data from several chunks
        """    
        accel_data    = query.get_sensor_data_from_filepath(filepath = filepath, sensor = "userAcceleration")
        rotation_data = query.get_sensor_data_from_filepath(filepath = filepath, sensor = "rotationRate")
        # if time series is not dataframe return as #ERROR # 
        if not ((isinstance(rotation_data, pd.DataFrame) and isinstance(accel_data, pd.DataFrame))):
            return "#ERROR"

        ## resample signals ##
        rotation_data = self.gait_processor.resample_signal(rotation_data)
        accel_data = self.gait_processor.resample_signal(accel_data)
        rotation_occurence = self.compute_rotational_features(accel_data, rotation_data)
        
        ## calculate gait features with rotation data removed ##
        gait_dict = self.split_dataframe_to_dict_chunk_by_interval(accel_data, rotation_occurence)
        
        ## calculate gait features per window iteration ##
        gait_feature_arr = []
        for chunks in gait_dict.keys():
            gait_feature_arr.append(self.compute_pdkit_feature_per_window(data = gait_dict[chunks]))
        gait_feature_arr = [j for i in gait_feature_arr for j in i]

        ## if no feature is appended to the chunk return as #ERROR ##
        if len(gait_feature_arr) == 0:
            return "#ERROR"
        return gait_feature_arr

        
    def rotation_feature_pipeline(self, filepath):
        """
        Function for featurizing rotation pipeline, which consist of features from pdkit package 
        that has been generated as a list on the rotation sequence 
        
        Args:
            Filepath    (str) : string of filepath to .synapseCache 
        
        Returns:
            RType: List
            Return combined list of walking data from several chunks
        """  
        rotation_data = query.get_sensor_data_from_filepath(filepath = filepath, sensor = "rotationRate")
        accel_data = query.get_sensor_data_from_filepath(filepath = filepath, sensor = "userAcceleration")   
        
        
        ## check if time series is of type dataframe                       
        if not ((isinstance(rotation_data, pd.DataFrame) and isinstance(accel_data, pd.DataFrame))):
            return "#ERROR"

        rotation_data = self.gait_processor.resample_signal(rotation_data)
        accel_data = self.gait_processor.resample_signal(accel_data)
        
        rotation_data = self.compute_rotational_features(accel_data, rotation_data)

        ## if no feature is appended to the rotation list return as #ERROR ##
        if len(rotation_data) == 0:
            return "#ERROR"
        return rotation_data

    def annotate_consecutive_zeros(self, data, feature):
        """
        TODO: has not been implemented to model, work in progress
        Function to annotate consecutive zeros in a column features in a pd.DataFrame,
        which can be used to as additional filter 
        Args:
            data (pd.DataFrame): A pandas dataframe 
            feature       (str): feature to assess on counting consecutive zeros
        
        returns:
            A new column-series of data with counted consecutive zeros (if available)
        """
        step_shift_measure = data[feature].ne(data[feature].shift()).cumsum()
        counts = data.groupby(['recordId', step_shift_measure])[feature].transform('size')
        data['consec_zero_steps_count'] = np.where(data[feature].eq(0), counts, 0)
        return data

    def featurize_wrapper(self, data):
        """
        Multiprocessing wrapper function for multiprocessing jobs (walking/balance)
        
        Args:
            data (pd.DataFrame): takes in pd.DataFrame of pathfile to synapseCache
        
        Returns:
            returns a list of json file featurized walking data on features column
        """
        data["gait.walk_features"] = data["gait.json_pathfile"].apply(self.gait_feature_pipeline)
        data["gait.rotation_features"] = data["gait.json_pathfile"].apply(self.rotation_feature_pipeline)
        return data


