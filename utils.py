import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Lambda, Input, Dense
from tensorflow.keras.models import Model

import numpy as np
import pandas as pd 
import random
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_prominences
import warnings
import time, copy
import os
import math

def distance(data, window_size):
    """
    Calculates distance (dissimilarity measure) between features
    
    Args:
        data: array of of learned features of size (nr. of windows) x (number of shared features)
        window_size: window size used for CPD
        
    Returns:
        Array of dissimilarities of size ((nr. of windows)-stride)
    """
    
    nr_windows = np.shape(data)[0]
    
    index_1 = range(window_size,nr_windows,1)
    index_2 = range(0,nr_windows-window_size,1)
    
    return np.sqrt(np.sum(np.square(data[index_1]-data[index_2]),1))

def parameters_to_cps(parameters,window_size):
    """
    Preparation for plotting ground-truth change points
    
    Args:
        parameters: array parameters used to generate time series, size Tx(nr. of parameters)
        window_size: window size used for CPD
        
    Returns:
        Array of which entry is non-zero in the presence of a change point. Higher values correspond to larger parameter changes.
    """
    
    length_ts = np.size(parameters,0)
        
    index1 = range(window_size-1,length_ts-window_size,1) #selects parameter at LAST time stamp of window
    index2 = range(window_size,length_ts-window_size+1,1) #selects parameter at FIRST time stamp of next window

    diff_parameters = np.sqrt(np.sum(np.square(parameters[index1]-parameters[index2]),1))
    
    max_diff = np.max(diff_parameters)
    
    return diff_parameters/max_diff

def cp_to_timestamps(changepoints, tolerance, length_ts):
    """
    Extracts time stamps of change points
    
    Args:
        changepoints:
        tolerance:
        length_ts: length of original time series
        
    Returns:
        list where each entry is a list with the windows affected by a change point
    """
    
    # if change points are consecutive, agg them into one list.
    # Ex [[79,80,81], [179], [200]]
    locations_cp = [idx for idx, val in enumerate(changepoints) if val > 0.0]
    
    output = []
    while len(locations_cp)>0:
        k = 0
        for i in range(len(locations_cp)-1):
            if  locations_cp[i]+1 == locations_cp[i+1]:
                k+=1
            else:
                break
        
        output.append(list(range(max(locations_cp[0]-tolerance,0),min(locations_cp[k]+1+tolerance,length_ts),1)))
        del locations_cp[:k+1]
        
    return output

def ts_to_windows(ts, onset, window_size, stride, normalization="timeseries"):
    """Transforms time series into list of windows"""
    windows = []
    len_ts = len(ts)
    onsets = range(onset, len_ts-window_size+1, stride)
    
    if normalization =="timeseries":
        for timestamp in onsets:
            windows.append(ts[timestamp:timestamp+window_size])
    elif normalization=="window":
        for timestamp in onsets:
            windows.append(np.array(ts[timestamp:timestamp+window_size])-np.mean(ts[timestamp:timestamp+window_size]))
            
    return np.array(windows)

def combine_ts(list_of_windows):
    """
    Combines a list of windows from multiple views to one list of windows
    
    Args:
        list_of_windows: list of windows from multiple views
        
    Returns:
        one array with the concatenated windows
    """
    nr_ts, nr_windows, window_size = np.shape(list_of_windows)
    tss = np.array(list_of_windows)
    new_ts = []
    for i in range(nr_windows):
        new_ts.append(tss[:,i,:].flatten())
    return np.array(new_ts)

def new_peak_prominences(distances):
    """
    Adapted calculation of prominence of peaks, based on the original scipy code
    
    Args:
        distances: dissimarity scores
    Returns:
        prominence scores
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        all_peak_prom = peak_prominences(distances,range(len(distances)))

    # add more filter
    # all_peak_prom = matched_filter(all_peak_prom[0], window_size=100)
    # all_peak_prom = peak_prominences(all_peak_prom, range(len(distances)))

    return all_peak_prom

def tpr_fpr(bps,distances, method="prominence",tol_dist=0):
    """Calculation of TPR and FPR
    
    Args:
        bps: list of breakpoints (change points)
        distances: list of dissimilarity scores
        method: prominence- or height-based change point score
        tol_dist: toleration distance
        
    Returns:
        list of TPRs and FPRs for different values of the detection threshold
    """
    
    peaks = find_peaks(distances)[0]
    peaks_prom = peak_prominences(distances,peaks)[0]

    # if there is no peak
    if len(peaks_prom) == 0:
        return [np.nan], [np.nan]

    peaks_prom_all = np.array(new_peak_prominences(distances)[0])
    
    distances = np.array(distances)
        
    bps = np.array(bps)
        
    if method == "prominence":
        
        nr_bps = len(bps)
        
        index_closest_peak = [0]*nr_bps
        bp_detected = [0]*nr_bps
        
        #determine for each bp the allowed range s.t. alarm is closest to bp
        
        ranges = [0] * nr_bps
        
        for i in range(nr_bps):
            
            if i==0:
                left = 0
            else:
                left = right
            if i==(nr_bps-1):
                right = len(distances)
            else:
                right = int((bps[i][-1]+bps[i+1][0])/2)+1
                
            ranges[i] = [left,right]
        
        # quantiles is threshold t in paper page 5
        quantiles = np.quantile(peaks_prom, np.array(range(51))/ 50)
        quantiles_set = set(quantiles)
        quantiles_set.add(0.)
        quantiles = list(quantiles_set)
        quantiles.sort()
        # print(f'quantiles: {quantiles}')
        
        nr_quant = len(quantiles)
        
        ncr = [0.]*nr_quant
        nal = [0.]*nr_quant
        
        for i in range(nr_quant):
            for j in range(nr_bps):
                
                bp_nbhd = peaks_prom_all[max(bps[j][0]-tol_dist,ranges[j][0]):min(bps[j][-1]+tol_dist+1,ranges[j][1])]
                
                if len(bp_nbhd) > 0 and max(bp_nbhd) >= quantiles[i]:
                    ncr[i] += 1
                            
            indices_all = np.array(range(len(distances)))
            heights_alarms = distances[peaks_prom_all>=quantiles[i]]
            nal[i] = len(heights_alarms)
                    
        ncr = np.array(ncr)
        nal = np.array(nal)
        # print(f'ncr: {ncr}')
        # print(f'nal: {nal}')
        '''
        NGT denotes the number of ground-truth change points, 
        NAL denotes the number of all detection alarms by the algorithm and 
        NCR is the number of times a ground-truth change point is correctly detected.
        '''
        ngt = nr_bps
        
        tpr = ncr/ngt
        fpr = (nal-ncr)/nal
        
        tpr = list(tpr[nal!=0])
        fpr = list(fpr[nal!=0])
        
        tpr.insert(0,1.0)
        fpr.insert(0,1.0)
        tpr.insert(len(tpr),0.0)
        fpr.insert(len(fpr),0.0)
    return tpr, fpr

def tpr_fpr_v2(bps,prominence, method="prominence",tol_dist=0):
    """Calculation of TPR and FPR
    
    Args:
        bps: list of breakpoints (change points)
        distances: list of dissimilarity scores
        method: prominence- or height-based change point score
        tol_dist: toleration distance
        
    Returns:
        list of TPRs and FPRs for different values of the detection threshold
    """
    peaks_prom_all = prominence

    peaks = find_peaks(prominence)[0]
    print(f'peaks length: {len(peaks)}')
    print(f'number elements > 0: {len(np.where(prominence > 0)[0])}')
    if len(peaks) == 0:
        print(f'there is no peak in prominence')
        print(f'len(prominence): {len(prominence)}, number elements > 0: {len(np.where(prominence > 0)[0])}')
        print(np.where(prominence > 0)[0])
        return 
    peaks_prom = prominence[peaks]

    # peaks_prom_all = np.array(new_peak_prominences(distances)[0])
    # distances = np.array(distances)
        
    bps = np.array(bps)
        
    if method == "prominence":
        
        nr_bps = len(bps)
        
        index_closest_peak = [0]*nr_bps
        bp_detected = [0]*nr_bps
        
        #determine for each bp the allowed range s.t. alarm is closest to bp
        
        ranges = [0] * nr_bps
        
        for i in range(nr_bps):
            
            if i==0:
                left = 0
            else:
                left = right
            if i==(nr_bps-1):
                right = len(peaks_prom_all)
            else:
                right = int((bps[i][-1]+bps[i+1][0])/2)+1
                
            ranges[i] = [left,right]
        
        # quantiles is threshold t in paper page 5
        quantiles = np.quantile(peaks_prom, np.array(range(51))/ 50)
        quantiles_set = set(quantiles)
        quantiles_set.add(0.)
        quantiles = list(quantiles_set)
        quantiles.sort()
        # print(f'quantiles: {quantiles}')
        
        nr_quant = len(quantiles)
        
        ncr = [0.]*nr_quant
        nal = [0.]*nr_quant
        
        for i in range(nr_quant):
            for j in range(nr_bps):
                
                bp_nbhd = peaks_prom_all[max(bps[j][0]-tol_dist,ranges[j][0]):min(bps[j][-1]+tol_dist+1,ranges[j][1])]
                
                if len(bp_nbhd) > 0 and max(bp_nbhd) >= quantiles[i]:
                    ncr[i] += 1
                            
            indices_all = np.array(range(len(peaks_prom_all)))
            # heights_alarms = distances[peaks_prom_all>=quantiles[i]]
            heights_alarms = peaks_prom_all>=quantiles[i]
            nal[i] = sum(heights_alarms)
                    
        ncr = np.array(ncr)
        nal = np.array(nal)
        # print(f'ncr: {ncr}')
        # print(f'nal: {nal}')
        '''
        NGT denotes the number of ground-truth change points, 
        NAL denotes the number of all detection alarms by the algorithm and 
        NCR is the number of times a ground-truth change point is correctly detected.
        '''
        ngt = nr_bps
        
        tpr = ncr/ngt
        fpr = (nal-ncr)/nal
        
        tpr = list(tpr[nal!=0])
        fpr = list(fpr[nal!=0])
        
        tpr.insert(0,1.0)
        fpr.insert(0,1.0)
        tpr.insert(len(tpr),0.0)
        fpr.insert(len(fpr),0.0)
    return tpr, fpr

def precision_recall_f1(bps,distances, method="prominence",tol_dist=0):
    """Calculation of Precision, Recall and F1
    
    Args:
        bps: list of breakpoints (change points)
        distances: list of dissimilarity scores
        method: prominence- or height-based change point score
        tol_dist: toleration distance
        
    Returns:
        list of TPRs and FPRs for different values of the detection threshold
    """
    
    peaks = find_peaks(distances)[0]
    peaks_prom = peak_prominences(distances,peaks)[0]
    # if there is no peak
    if len(peaks_prom) == 0:
        return [np.nan], [np.nan], [np.nan], [np.nan]

    peaks_prom_all = np.array(new_peak_prominences(distances)[0])

    distances = np.array(distances)
        
    bps = np.array(bps)
        
    if method == "prominence":
        
        nr_bps = len(bps)
        
        index_closest_peak = [0]*nr_bps
        bp_detected = [0]*nr_bps
        
        #determine for each bp the allowed range s.t. alarm is closest to bp
        
        ranges = [0] * nr_bps
        
        for i in range(nr_bps):
            
            if i==0:
                left = 0
            else:
                left = right
            if i==(nr_bps-1):
                right = len(distances)
            else:
                right = int((bps[i][-1]+bps[i+1][0])/2)+1
                
            ranges[i] = [left,right]
        
        # quantiles is threshold t in paper page 5
        quantiles = np.quantile(peaks_prom, np.array(range(21))/ 20)
        quantiles_set = set(quantiles)
        quantiles_set.add(0.)
        quantiles = list(quantiles_set)
        quantiles.sort()
        # print(f'quantiles: {quantiles}')
        
        nr_quant = len(quantiles)
        
        ncr = [0.]*nr_quant
        nal = [0.]*nr_quant
        
        
        for i in range(nr_quant):
            for j in range(nr_bps):
                
                bp_nbhd = peaks_prom_all[max(bps[j][0]-tol_dist,ranges[j][0]):min(bps[j][-1]+tol_dist+1,ranges[j][1])]
                
                if len(bp_nbhd) > 0 and max(bp_nbhd) >= quantiles[i]:
                    ncr[i] += 1
                            
            indices_all = np.array(range(len(distances)))
            heights_alarms = distances[peaks_prom_all>=quantiles[i]]
            nal[i] = len(heights_alarms)
                    
        ncr = np.array(ncr)
        nal = np.array(nal)
  
        
        ngt = nr_bps
        
        recall = ncr/ngt
        precision = ncr/nal
        # print(f'precision: {precision}')
        # print(f'recall: {recall}')
        f1 = 2 * precision * recall / (precision + recall)
        # print(f'f1: {f1}')
        
       
    return precision, recall, f1, quantiles

def precision_recall_f1_v2(bps, prominence, method="prominence",tol_dist=0):
    """Calculation of Precision, Recall and F1
    
    Args:
        bps: list of breakpoints (change points)
        distances: list of dissimilarity scores
        method: prominence- or height-based change point score
        tol_dist: toleration distance
        
    Returns:
        list of TPRs and FPRs for different values of the detection threshold
    """
    
    peaks = find_peaks(prominence)[0]
    peaks_prom_all = prominence

    if len(peaks) == 0:
        print(f'there is no peak in prominence')
        print(f'len(prominence): {len(prominence)}, number elements > 0: {len(np.where(prominence > 0)[0])}')
        print(np.where(prominence > 0)[0])
        return 
    peaks_prom = prominence[peaks]

    distances = np.array(distances)
        
    bps = np.array(bps)
        
    if method == "prominence":
        
        nr_bps = len(bps)
        
        index_closest_peak = [0]*nr_bps
        bp_detected = [0]*nr_bps
        
        #determine for each bp the allowed range s.t. alarm is closest to bp
        
        ranges = [0] * nr_bps
        
        for i in range(nr_bps):
            
            if i==0:
                left = 0
            else:
                left = right
            if i==(nr_bps-1):
                right = len(peaks_prom_all)
            else:
                right = int((bps[i][-1]+bps[i+1][0])/2)+1
                
            ranges[i] = [left,right]
        
        # quantiles is threshold t in paper page 5
        quantiles = np.quantile(peaks_prom, np.array(range(51))/ 50)
        quantiles_set = set(quantiles)
        quantiles_set.add(0.)
        quantiles = list(quantiles_set)
        quantiles.sort()
        # print(f'quantiles: {quantiles}')
        
        nr_quant = len(quantiles)
        
        ncr = [0.]*nr_quant
        nal = [0.]*nr_quant
        
        
        for i in range(nr_quant):
            for j in range(nr_bps):
                
                bp_nbhd = peaks_prom_all[max(bps[j][0]-tol_dist,ranges[j][0]):min(bps[j][-1]+tol_dist+1,ranges[j][1])]
                
                if len(bp_nbhd) > 0 and max(bp_nbhd) >= quantiles[i]:
                    ncr[i] += 1
                            
            indices_all = np.array(range(len(distances)))
            heights_alarms = distances[peaks_prom_all>=quantiles[i]]
            nal[i] = len(heights_alarms)
                    
        ncr = np.array(ncr)
        nal = np.array(nal)
  
        
        ngt = nr_bps
        
        precision = ncr/ngt
        recall = ncr/nal

        f1 = 2 * precision * recall / (precision + recall)
       
    return precision, recall, f1, quantiles

def matched_filter(signal, window_size):
    """
    Matched filter for dissimilarity measure smoothing (and zero-delay weighted moving average filter for shared feature smoothing)
    
    Args:
        signal: input signal
        window_size: window size used for CPD
    Returns:
        filtered signal
    """
    mask = np.ones((2*window_size+1,))
    for i in range(window_size):
        mask[i] = i/(window_size**2)
        mask[-(i+1)] = i/(window_size**2)
    mask[window_size] = window_size/(window_size**2)
        
    signal_out = np.zeros(np.shape(signal))
    
    if len(np.shape(signal)) >1:
        for i in range(np.shape(signal)[1]):
            signal_extended = np.concatenate((signal[0,i]*np.ones(window_size), signal[:,i], signal[-1,i]*np.ones(window_size)))
            signal_out[:,i] = np.convolve(signal_extended, mask, 'valid')
    else:
        signal = np.concatenate((signal[0]*np.ones(window_size), signal, signal[-1]*np.ones(window_size)))
        signal_out = np.convolve(signal, mask, 'valid')
    
    return signal_out

def minmaxscale(data, a, b):
    """
    Scales data to the interval [a,b]
    """
    data_min = np.amin(data)
    data_max = np.amax(data)
        
    return (b-a)*(data-data_min)/(data_max-data_min)+a

def calc_fft(windows, nfft=30, norm_mode="timeseries"):
    """
    Calculates the DFT for each window and transforms its length
    
    Args:
        windows: time series windows
        nfft: number of points used for the calculation of the DFT
        norm_mode: ensure that the timeseries / each window has zero mean
        
    Returns:
        frequency domain windows, each window having size nfft//2 (+1 for timeseries normalization)
    """
    mean_per_segment = np.mean(windows, axis=-1)
    mean_all = np.mean(mean_per_segment, axis=0)
    
    if norm_mode == "window":
        windows = windows-mean_per_segment[:,None]
        windows_fft = abs(np.fft.fft(windows, nfft))[...,1:nfft//2+1]
    elif norm_mode == "timeseries":
        windows = windows-mean_all
        windows_fft = abs(np.fft.fft(windows, nfft))[...,:nfft//2+1]
        
    
    fft_max = np.amax(windows_fft)
    fft_min = np.amin(windows_fft)
        
    windows_fft = 2*(windows_fft-fft_min)/(fft_max-fft_min)-1
    
    return windows_fft

def plot_cp(distances, time_start, time_stop, plot_prominences=False, breakpoints=None):
    """
    Plots dissimilarity measure with ground-truth changepoints
    
    Args:
        distances: dissimilarity measures
        parameters: array parameters used to generate time series, size Tx(nr. of parameters)
        window_size: window size used for CPD
        time_start: first time stamp of plot
        time_stop: last time stamp of plot
        plot_prominences: True/False
        
    Returns:
        plot of dissimilarity measure with ground-truth changepoints
    
    """
    # this is ground truth
    # however, this code is not produce ground truth. It assumes ground-truth for example purpose
    # breakpoints = parameters_to_cps(parameters,window_size)


    length_ts = np.size(breakpoints)
    t = range(len(distances))


    x = t
    z = distances
    y = breakpoints#[:,0]

    peaks = find_peaks(distances)[0] # just find peak based on neighbour point
    peaks_prom = peak_prominences(distances,peaks)[0]
    peaks_prom_all = np.array(new_peak_prominences(distances)[0])
    # print(f'peaks prom all. shape: {peaks_prom_all.shape}, values: {peaks_prom_all[:4]}')
    fig, ax = plt.subplots(figsize=(15,4))
    ax.plot(x,z,color="black")
    
    if plot_prominences:
        ax.plot(x,peaks_prom_all, color="blue")

    ax.set_xlim(time_start,time_stop)
    ax.set_ylim(0,1.5*max(z))
    plt.xlabel("time")
    plt.ylabel("dissimilarity")

    ax.plot(peaks,distances[peaks],'go') # peak point

    height_line = 1

    # ax.fill_between(x, 0, height_line, where=y > 0.0001,
    #             color='red', alpha=0.2, transform=ax.get_xaxis_transform())
    # ax.fill_between(x, 0, height_line, where=y > 0.25,
    #             color='red', alpha=0.2, transform=ax.get_xaxis_transform())
    # ax.fill_between(x, 0, height_line, where=y > 0.5,
    #             color='red', alpha=0.2, transform=ax.get_xaxis_transform())
    # ax.fill_between(x, 0, height_line, where=y > 0.75,
    #             color='red', alpha=0.2, transform=ax.get_xaxis_transform())
    ax.fill_between(x, 0, height_line, where=y > 0,
                color='red', alpha=0.2, transform=ax.get_xaxis_transform())
    plt.show()
    
def get_auc(distances, tol_distances, breakpoints, is_plot=True):
    """
    Calculation of AUC for toleration distances in range(TD_start, TD_stop, TD_step) + plot of corresponding ROC curves
    
    Args:
        distances: dissimilarity measures
        tol_distances: list of different toleration distances
        parameters: array parameters used to generate time series, size Tx(nr. of parameters)
        window_size: window size used for CPD
        breakpoints: change point ground truth. 
        
    Returns:
        list of AUCs for every toleration distance
    """
    
    # breakpoints = parameters_to_cps(parameters,window_size)

    legend = []
    # neighbour breakpoint will be aggregate into the same timestamp
    list_of_lists = cp_to_timestamps(breakpoints,0,np.size(breakpoints))
    auc = []

    # what is tolerence distance?
    for i in tol_distances:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            tpr, fpr = tpr_fpr(list_of_lists,distances, "prominence",i)
        if is_plot:
            plt.plot(fpr,tpr)
            legend.append("tol. dist. = "+str(i))
        auc.append(np.abs(np.trapz(tpr,x=fpr)))

    # print(auc)
    if is_plot:
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title("ROC curve")
        plt.plot([0,1],[0,1],'--')
        legend.append("TPR=FPR")
        plt.legend(legend)
        plt.show()

        plt.plot(tol_distances,auc)
        plt.xlabel("toleration distance")
        plt.ylabel("AUC")
        plt.title("AUC")
        plt.show()
    
    return auc

def get_F1(distances, tol_distances, breakpoints, is_plot=True):
    """
    Calculation of AUC for toleration distances in range(TD_start, TD_stop, TD_step) + plot of corresponding ROC curves
    
    Args:
        distances: dissimilarity measures
        tol_distances: list of different toleration distances
        parameters: array parameters used to generate time series, size Tx(nr. of parameters)
        window_size: window size used for CPD
        breakpoints: change point ground truth. 
        
    Returns:
        list of AUCs for every toleration distance
    """
    
    # breakpoints = parameters_to_cps(parameters,window_size)

    legend = []
    # neighbour breakpoint will be aggregate into the same timestamp
    list_of_lists = cp_to_timestamps(breakpoints,0,np.size(breakpoints))
    f1s = []

    # what is tolerence distance?
    for i in tol_distances:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            precision, recall, f1, quantiles = precision_recall_f1(list_of_lists,distances, "prominence",i)
            f1s.append(f1)

            if is_plot:
                plt.plot(quantiles, f1)
                legend.append("tol. dist. = "+str(i))
    
    if is_plot:
        plt.xlabel("Quantile threshold")
        plt.ylabel("F1")
        plt.legend(legend)
        plt.show()

    # f1max = []
    # for i in range(len(tol_distances)):
    #     f1max.append(max(f1s[i]))
    # print(f'f1 max: {f1max}')
    return f1s

def get_F1_v2(prominence, tol_distances, breakpoints, is_plot=True):
    """
    Calculation of AUC for toleration distances in range(TD_start, TD_stop, TD_step) + plot of corresponding ROC curves
    
    Args:
        distances: dissimilarity measures
        tol_distances: list of different toleration distances
        parameters: array parameters used to generate time series, size Tx(nr. of parameters)
        window_size: window size used for CPD
        breakpoints: change point ground truth. 
        
    Returns:
        list of AUCs for every toleration distance
    """
    
    # breakpoints = parameters_to_cps(parameters,window_size)

    legend = []
    # neighbour breakpoint will be aggregate into the same timestamp
    list_of_lists = cp_to_timestamps(breakpoints,0,np.size(breakpoints))
    f1s = []

    # what is tolerence distance?
    for i in tol_distances:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            precision, recall, f1, quantiles = precision_recall_f1(list_of_lists, prominence, "prominence",i)
            f1s.append(f1)
            
            if is_plot:
                plt.plot(quantiles, f1)
                legend.append("tol. dist. = "+str(i))
    
    if is_plot:
        plt.xlabel("Quantile threshold")
        plt.ylabel("F1")
        plt.legend(legend)
        plt.show()

    f1max = []
    for i in range(len(tol_distances)):
        f1max.append(max(f1s[i]))
    print(f'f1 max: {f1max}')
    return f1s

def get_auc_v2(prominence, tol_distances, breakpoints, is_plot=True):
    """
    Calculation of AUC for toleration distances in range(TD_start, TD_stop, TD_step) + plot of corresponding ROC curves
    
    Args:
        prominence: all prominence of dissimilarity
        tol_distances: list of different toleration distances
        parameters: array parameters used to generate time series, size Tx(nr. of parameters)
        window_size: window size used for CPD
        breakpoints: change point ground truth. 
        
    Returns:
        list of AUCs for every toleration distance
    """
    
    # breakpoints = parameters_to_cps(parameters,window_size)

    legend = []
    # neighbour breakpoint will be aggregate into the same timestamp
    list_of_lists = cp_to_timestamps(breakpoints,0,np.size(breakpoints))
    auc = []

    # what is tolerence distance?
    for i in tol_distances:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            tpr, fpr = tpr_fpr_v2(list_of_lists,prominence, "prominence",i)
        
        if is_plot:
            plt.plot(fpr,tpr)
            legend.append("tol. dist. = "+str(i))
        auc.append(np.abs(np.trapz(tpr,x=fpr)))

    print(auc)
    if is_plot:
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title("ROC curve")
        plt.plot([0,1],[0,1],'--')
        legend.append("TPR=FPR")
        plt.legend(legend)
        plt.show()

    # plt.plot(tol_distances,auc)
    # plt.xlabel("toleration distance")
    # plt.ylabel("AUC")
    # plt.title("AUC")
    # plt.show()
    
    return auc

def create_folder_if_not_exist(folder_name):
    if os.path.exists(folder_name) == False:
        os.makedirs(folder_name)

def create_prominence_from_multi_channels(num_channels:int, dissimilarities, window_size:int):
    # get prominence for each channel 
    prominence_all = []
    for channel in range(num_channels):
        prominence_all.append(new_peak_prominences(dissimilarities[channel])[0])

    distance = np.array(prominence_all)
    dis_threshold = np.mean(distance, axis = 1) # percentige for each axis 
    distance_len = len(distance[0])
    new_distance = np.zeros((distance_len)) # after combined change point accross channels
    i = 0
    while i < distance_len:
        potential_indexes = np.where(distance[:, i] > dis_threshold)[0]
        if len(potential_indexes) == 0:
            i += 1
            continue
        anchor_index = potential_indexes[0] # first chanel have 
        left_index, right_index = max(0, i), min(i + window_size, distance_len - 1)
        tmp = []
        for channel in range(num_channels):
            channel_distance = distance[channel][left_index: right_index]
            searching = np.where(channel_distance > dis_threshold[channel])[0]
            if len(searching) > 0:
                mean_index = math.floor(searching.mean())
                mean_value = channel_distance[searching].mean()

                tmp.append([mean_index, mean_value])
        tmp = np.array(tmp)
        if len(tmp) == num_channels: # hyperparameter
            # if the number of channel have change point <= num_channels/2 => ignore current index
            tmp = tmp.T # (channel = row, column = value)
            tmp_mean = tmp.mean(axis=1)
            new_distance[math.floor(left_index + tmp_mean[0])] = tmp_mean[1]

        # update i
        i += window_size
    return new_distance


def setup_random_seed():
    seed_value= 42

    # 1. Set the `PYTHONHASHSEED` environment variable at a fixed value
    import os
    os.environ['PYTHONHASHSEED']=str(seed_value)

    # 2. Set the `python` built-in pseudo-random generator at a fixed value
    import random
    random.seed(seed_value)

    # 3. Set the `numpy` pseudo-random generator at a fixed value
    import numpy as np
    np.random.seed(seed_value)

    # 4. Set the `tensorflow` pseudo-random generator at a fixed value
    import tensorflow as tf
    tf.random.set_seed(seed_value)

def plot_channels(raw_df: pd.DataFrame, column_data_str: str, labels: np.array):
    f, ax = plt.subplots(len(column_data_str), 1, figsize=(30, 4 * len(column_data_str)), squeeze=False)
    f.tight_layout(pad=2)
    timeseries_len = raw_df.shape[0]
    for index, column_name in enumerate(column_data_str):
        ax[index, 0].plot(range(timeseries_len), raw_df[column_name])
        ax[index, 0].set_title(f"channel: {column_name}", fontsize=20)

        height_line = 1
        ax[index, 0].fill_between(range(timeseries_len), 0, height_line, where=labels > 0, color='red', alpha=0.2, transform=ax[index, 0].get_xaxis_transform())
    plt.show()