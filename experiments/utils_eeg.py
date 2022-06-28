import pandas as pd 
import numpy as np 

def create_break_point_index(labels_df: pd.DataFrame):
    change_event_index = labels_df.max(axis=1).to_numpy()
    possitive_index = np.where(np.array(change_event_index) > 0)[0]
    change_index = []
    i = 0
    while i < len(possitive_index) - 1:
        change_index.append(possitive_index[i])
        while i + 1 < len(possitive_index) and possitive_index[i] + 1 == possitive_index[i + 1]: 
            i += 1
        if i + 1 < len(possitive_index):
            change_index.append(possitive_index[i] + 1)
        i += 1
    result = np.array([0] * len(change_event_index))
    result[change_index] = [1] * len(change_index)
    return result