import pandas as pd
import numpy as np
from control import DMDc
from scipy.stats import entropy
from numpy import linalg as LA
import json
import os

FPS = 29.97

def get_angle(pose, x, y):
    """
    Get the angle between two keypoints and the horizontal
    pose: a set of pose keypoints
    x: the number of the first keypoint (between 0 and 24)
    y: the number of the second openpose keypoint (between 0 and 24)
    """
    nose = x
    ear=y 
    # get angle between line from ear to nose, line from ear to horizontal
    nose_x = pose[str(nose*3)];  nose_y = pose[str(nose*3+1)]
    ear_x  = pose[str(ear*3)];   ear_y  = pose[str(ear*3+1)]
    
    nose_x = nose_x.mask(nose_x==0); nose_y = nose_y.mask(nose_y==0)
    ear_x = ear_x.mask(ear_x==0); ear_y = ear_y.mask(ear_y==0)
    
    h_x    = pose[str(ear*3)]+1; h_y    = pose[str(ear*3+1)]
    
    
    angle = np.arctan2(h_y - ear_y, h_x - ear_x) - np.arctan2(nose_y - ear_y, nose_x - ear_x);
    angle = angle*180/np.pi 
    angle.loc[angle < 0] = angle.loc[angle < 0] + 360
    return angle.mask(angle==0)

def adjust(signal):
    x = signal.copy()
    s = signal.copy().dropna()
    d300 = s.loc[s.diff()<-300].index.values
    d100 = s.loc[s.diff()>100].index.values
    for i in d300:
        j = d100[d100>i]
        if len(j)>0:
            j=j[0]
        else:
            j=s.index.values[-1]
        s.loc[i:j-1] = s.loc[i:j-1] + 360
        x.loc[i:j-1] = x.loc[i:j-1] + 360 
    d300 = s.loc[s.diff()>300].index.values
    d100 = s.loc[s.diff()<-100].index.values
    for i in d300:
        j = d100[d100>i]
        if len(j)>0:
            j=j[0]
        else:
            j=s.index.values[-1]
        s.loc[i:j-1] = s.loc[i:j-1] - 360
        x.loc[i:j-1] = x.loc[i:j-1] - 360
    return x

def num_to_str(num):
    '''Convert number to string, adding a leading 0 if num<10'''
    if num < 10:
        return '0' + str(num)
    return str(num)

def get_visit(iid, age):
    '''Return a visit id based on infant number and age'''
    return 'p' + num_to_str(iid) + num_to_str(age)

def time_to_frame(time, fps):
    '''Convert a m:ss:ms time string to frame number'''
    minutes = int(time[0])
    seconds = int(time[2:4])
    ms = int(time[5:7])
    return int((minutes*60 + seconds + ms/100)*fps)

def get_times(iid, age):
    ''' Returns the times of each stage in frames '''
    time_map = pd.read_csv('map.csv')
    row = time_map.loc[time_map.visit==get_visit(iid,age)]
    times = { 
        'start_play': time_to_frame(row.start_play.values[0], FPS),
        'start_sf': time_to_frame(row.start_sf.values[0], FPS),
        'end_sf': time_to_frame(row.end_sf.values[0], FPS),
        'end_reunion': time_to_frame(row.end_play.values[0], FPS)
    }
    return times

def get_op(iid, age):
    person_numbers = pd.read_csv('person_number.csv')
    row = person_numbers.loc[(person_numbers.Infant==iid) & (person_numbers.Month==age)]
    mom_number = int(row['Mom number'].values[0])
    infant_number = int(row['Infant number'].values[0])
    visit = get_visit(iid, age)
    mom_df = pd.read_csv(f'data/{visit}/person{mom_number}.csv')
    infant_df = pd.read_csv(f'data/{visit}/person{infant_number}.csv')
    mom_df = mom_df.mask(mom_df==0).interpolate(method='linear')
    mom_df = mom_df.interpolate(method='bfill').interpolate(method='ffill')
    infant_df = infant_df.mask(infant_df==0).interpolate(method='linear')
    infant_df = infant_df.interpolate(method='bfill').interpolate(method='ffill')
    mom_arm, infant_arm = get_arm_features_unmasked(iid, age)
    mom_df['head_x'] = (mom_df['0']-mom_df['3'])
    mom_df['head_y'] = (mom_df['1']-mom_df['4'])
    infant_df['head_x'] = (infant_df['0']-infant_df['3'])
    infant_df['head_y'] = (infant_df['1']-infant_df['4'])
    mom_elbow_x, infant_elbow_x = str(mom_arm[1]*3), str(infant_arm[1]*3)
    mom_elbow_y, infant_elbow_y =  str(mom_arm[1]*3+1), str(infant_arm[1]*3+1)
    mom_df['arm_x'] = (mom_df[mom_elbow_x]-mom_df['3'])
    mom_df['arm_y'] = (mom_df[mom_elbow_y]-mom_df['4'])
    infant_df['arm_x'] = (infant_df[infant_elbow_x]-infant_df['3'])
    infant_df['arm_y'] = (infant_df[infant_elbow_y]-infant_df['4'])
    return mom_df, infant_df

def get_f0(iid, age):
    i = num_to_str(iid)
    age = num_to_str(age)
    mom_folder = 'mother-infant-interactions/F0_csv_mom'
    infant_folder = 'mother-infant-interactions/F0_csv_infant'
    mom_df = pd.read_csv(f'{mom_folder}/t{i}{age}')
    mom_df.loc[mom_df.pitch.isna(),'pitch']=0
    infant_df = pd.read_csv(f'{infant_folder}/t{i}{age}')
    infant_df.loc[infant_df.pitch.isna(),'pitch']=0
    # print(mom_df)
    return mom_df, infant_df

def fill_gaps(df, mi):
    # if two values are less than 15 apart, fill the gaps
    idx = df.loc[df[mi]>0].index.values
    for i in range(idx.shape[0]-1):
        if idx[i+1]-idx[i]<16:
            df.loc[idx[i:i+1], mi] = 1
    return df
    

def get_op_unmasked(iid, age):
    person_numbers = pd.read_csv('person_number.csv')
    row = person_numbers.loc[(person_numbers.Infant==iid) & (person_numbers.Month==age)]
    mom_number = int(row['Mom number'].values[0])
    infant_number = int(row['Infant number'].values[0])
    visit = get_visit(iid, age)
    mom_df = pd.read_csv(f'data/{visit}/person{mom_number}.csv')
    infant_df = pd.read_csv(f'data/{visit}/person{infant_number}.csv')
    return mom_df, infant_df

def get_arm_features_unmasked(iid, age):
    '''Return the features of the arm with the higher confidence'''
    mom_df, infant_df = get_op_unmasked(iid, age)
    left = np.array([2, 3, 4])*3 +2
    right = np.array([5, 6, 7])*3 +2
    left_lm = [2,3,4]
    right_lm = [5,6,7]
    mom_left = mom_df.iloc[:,left].mean().mean()
    mom_right = mom_df.iloc[:,right].mean().mean()
    infant_left = infant_df.iloc[:,left].mean().mean()
    infant_right = infant_df.iloc[:,right].mean().mean()
    if mom_left > mom_right:
        mom_arm = left_lm
    else:
        mom_arm = right_lm
    if infant_left > infant_right:
        infant_arm = left_lm
    else:
        infant_arm = right_lm
    return mom_arm, infant_arm

def get_normalization_distance(iid, age):
    '''Get the average distance between the most frequently in-view eye and ear 
    as the normalization distance, and divide all values by this distance'''
    left = np.array([0, 17])*3 +2
    right = np.array([0, 18])*3 +2
    left_lm = np.array([0, 17])
    right_lm = np.array([0, 18])
    mom_df, infant_df = get_op(iid, age)
    mom_left = mom_df.iloc[:,left].mean().mean()
    mom_right = mom_df.iloc[:,right].mean().mean()
    infant_left = infant_df.iloc[:,left].mean().mean()
    infant_right = infant_df.iloc[:,right].mean().mean()
    if mom_left > mom_right:
        mom_face = left_lm
    else:
        mom_face = right_lm
    if infant_left > infant_right:
        infant_face = left_lm
    else:
        infant_face = right_lm
    # mom distance
    mom_df = mom_df[mom_df.iloc[:,mom_face].sum(axis=1)>0]
    x_dist = mom_df.iloc[:,mom_face[0]*3]-mom_df.iloc[:,mom_face[1]*3]
    y_dist = mom_df.iloc[:,mom_face[0]*3+1]-mom_df.iloc[:,mom_face[1]*3+1]
    mom_distance = np.sqrt(np.square(x_dist) + np.square(y_dist)).mean()

    # infant distance
    infant_df= infant_df[infant_df.iloc[:,infant_face].sum(axis=1)>0]
    x_dist = infant_df.iloc[:,infant_face[0]*3]-infant_df.iloc[:,infant_face[1]*3]
    y_dist = infant_df.iloc[:,infant_face[0]*3+1]-infant_df.iloc[:,infant_face[1]*3+1]
    infant_distance = np.sqrt(np.square(x_dist) + np.square(y_dist)).mean()
    return mom_distance, infant_distance   

def get_arm_features(iid, age):
    '''Return the features of the arm with the higher confidence'''
    mom_df, infant_df = get_op(iid, age)
    left = np.array([2, 3, 4])*3 +2
    right = np.array([5, 6, 7])*3 +2
    left_lm = [2,3,4]
    right_lm = [5,6,7]
    mom_left = mom_df.iloc[:,left].mean().mean()
    mom_right = mom_df.iloc[:,right].mean().mean()
    infant_left = infant_df.iloc[:,left].mean().mean()
    infant_right = infant_df.iloc[:,right].mean().mean()
    if mom_left > mom_right:
        mom_arm = left_lm
    else:
        mom_arm = right_lm
    if infant_left > infant_right:
        infant_arm = left_lm
    else:
        infant_arm = right_lm
    return mom_arm, infant_arm

def train_dmdc(iid, age, features, diff_features, stage):
    if stage not in ['play', 'sf', 'reunion']:
        raise Exception('stage must be one of [play, sf, reunion]')
    mom_df, infant_df = get_op(iid, age)
    # mom_dist, infant_dist = get_normalization_distance(iid, age)
    # mom_df = mom_df/mom_dist
    # infant_df = infant_df/infant_dist
    times = get_times(iid, age)
    if stage=='play':
        mom_stage = mom_df.iloc[times['start_play']:times['start_sf'],:]
        infant_stage = infant_df.iloc[times['start_play']:times['start_sf'],:]
    elif stage=='sf':
        mom_stage = mom_df.iloc[times['start_sf']:times['end_sf'],:]
        infant_stage = infant_df.iloc[times['start_sf']:times['end_sf'],:]
    else:
        mom_stage = mom_df.iloc[times['end_sf']:times['end_reunion'],:]
        infant_stage = infant_df.iloc[times['end_sf']:times['end_reunion'],:]

    mom_stage = np.hstack((mom_stage.loc[:,features], mom_stage.loc[:,diff_features].diff()))[1:,:]
    infant_stage = np.hstack((infant_stage.loc[:,features], infant_stage.loc[:,diff_features].diff()))[1:,:]

    X = mom_stage[:-1,:]
    Y = infant_stage[:-1,:]
    Z = mom_stage[1:,:]

    dmdc = DMDc(B=None)
    dmdc.fit(X,Y,Z)

    return dmdc.A, dmdc.B

def train_dmdc_asym(iid, age, mom_features, infant_features, stage):
    if stage not in ['play', 'sf', 'reunion']:
        raise Exception('stage must be one of [play, sf, reunion]')
    mom_df, infant_df = get_op(iid, age)
    # mom_dist, infant_dist = get_normalization_distance(iid, age)
    # mom_df = mom_df/mom_dist
    # infant_df = infant_df/infant_dist
    times = get_times(iid, age)
    if stage=='play':
        mom_stage = mom_df.iloc[times['start_play']:times['start_sf'],:]
        infant_stage = infant_df.iloc[times['start_play']:times['start_sf'],:]
    elif stage=='sf':
        mom_stage = mom_df.iloc[times['start_sf']:times['end_sf'],:]
        infant_stage = infant_df.iloc[times['start_sf']:times['end_sf'],:]
    else:
        mom_stage = mom_df.iloc[times['end_sf']:times['end_reunion'],:]
        infant_stage = infant_df.iloc[times['end_sf']:times['end_reunion'],:]
    mom_stage = mom_stage.loc[:,mom_features].values
    infant_stage = infant_stage.loc[:,infant_features].values
    # mom_stage = np.hstack((mom_stage.loc[:,mom_features], mom_stage.loc[:,mom_features].diff()))[1:,:]
    # infant_stage = np.hstack((infant_stage.loc[:,infant_features], infant_stage.loc[:,infant_features].diff()))[1:,:]

    X = mom_stage[:-1,:]
    Y = infant_stage[:-1,:]
    Z = mom_stage[1:,:]

    dmdc = DMDc(B=None)
    dmdc.fit(X,Y,Z)

    return dmdc.A, dmdc.B

def train_dmdc_audio(iid, age, stage):
    if stage not in ['play', 'sf', 'reunion']:
        raise Exception('stage must be one of [play, sf, reunion]')
    mom_df, infant_df = get_f0(iid, age)
    times = get_times(iid, age)
    if stage=='play':
        mom_stage = mom_df.iloc[times['start_play']:times['start_sf'],:]
        infant_stage = infant_df.iloc[times['start_play']:times['start_sf'],:]
    elif stage=='sf':
        mom_stage = mom_df.iloc[times['start_sf']:times['end_sf'],:]
        infant_stage = infant_df.iloc[times['start_sf']:times['end_sf'],:]
    else:
        mom_stage = mom_df.iloc[times['end_sf']:times['end_reunion'],:]
        infant_stage = infant_df.iloc[times['end_sf']:times['end_reunion'],:]
    mom_stage = mom_stage[['mom']].values
    infant_stage = infant_stage[['baby']].values


    # mom_stage = np.hstack((mom_stage.loc[:,mom_features], mom_stage.loc[:,mom_features].diff()))[1:,:]
    # infant_stage = np.hstack((infant_stage.loc[:,infant_features], infant_stage.loc[:,infant_features].diff()))[1:,:]

    X = mom_stage[:-1,:]
    Y = infant_stage[:-1,:]
    Z = mom_stage[1:,:]

    print(X.sum(), Y.sum(), Z.sum())

    dmdc = DMDc(B=None)
    dmdc.fit(X,Y,Z)

    return dmdc.A, dmdc.B


def train_dmdc_both(iid, age, mom_features, infant_features, stage):
    if stage not in ['play', 'sf', 'reunion']:
        raise Exception('stage must be one of [play, sf, reunion]')
    mom_df, infant_df = get_op(iid, age)
    mom_f0, infant_f0 = get_f0(iid, age)
    cols = np.concatenate([mom_df.columns.values, mom_f0.columns.values])
    mom_df = pd.concat([mom_df, mom_f0], ignore_index=True, axis=1)
    infant_df = pd.concat([infant_df, infant_f0], ignore_index=True, axis=1)

    # mom_features.append('mom')
    infant_features.append('baby')

    mom_df.columns=cols
    infant_df.columns=cols

    # mom_dist, infant_dist = get_normalization_distance(iid, age)
    # mom_df = mom_df/mom_dist
    # infant_df = infant_df/infant_dist
    times = get_times(iid, age)
    if stage=='play':
        mom_stage = mom_df.iloc[times['start_play']:times['start_sf'],:]
        infant_stage = infant_df.iloc[times['start_play']:times['start_sf'],:]
    elif stage=='sf':
        mom_stage = mom_df.iloc[times['start_sf']:times['end_sf'],:]
        infant_stage = infant_df.iloc[times['start_sf']:times['end_sf'],:]
    else:
        mom_stage = mom_df.iloc[times['end_sf']:times['end_reunion'],:]
        infant_stage = infant_df.iloc[times['end_sf']:times['end_reunion'],:]
    mom_stage = mom_stage.loc[:,mom_features].values
    infant_stage = infant_stage.loc[:,infant_features].values
    # mom_stage = np.hstack((mom_stage.loc[:,mom_features], mom_stage.loc[:,mom_features].diff()))[1:,:]
    # infant_stage = np.hstack((infant_stage.loc[:,infant_features], infant_stage.loc[:,infant_features].diff()))[1:,:]

    X = mom_stage[:-1,:]
    Y = infant_stage[:-1,:]
    Z = mom_stage[1:,:]

    dmdc = DMDc(B=None)
    dmdc.fit(X,Y,Z)

    return dmdc.A, dmdc.B


def train_dmdc_distances(iid, age, mom_features, infant_features, stage):
    if stage not in ['play', 'sf', 'reunion']:
        raise Exception('stage must be one of [play, sf, reunion]')
    mom_df, infant_df = get_op(iid, age)
    mom_arm, infant_arm = get_arm_features_unmasked(iid, age)

    mom_dist, infant_dist = get_normalization_distance(iid, age)
    mom_df = mom_df/mom_dist
    infant_df = infant_df/infant_dist

    mom_df['head_x'] = (mom_df['0']-mom_df['3'])
    mom_df['head_y'] = (mom_df['1']-mom_df['4'])
    infant_df['head_x'] = (infant_df['0']-infant_df['3'])
    infant_df['head_y'] = (infant_df['1']-infant_df['4'])
    mom_elbow_x, infant_elbow_x = str(mom_arm[1]*3), str(infant_arm[1]*3)
    mom_elbow_y, infant_elbow_y =  str(mom_arm[1]*3+1), str(infant_arm[1]*3+1)
    mom_df['arm_x'] = (mom_df[mom_elbow_x]-mom_df['3'])
    mom_df['arm_y'] = (mom_df[mom_elbow_y]-mom_df['4'])
    infant_df['arm_x'] = (infant_df[infant_elbow_x]-infant_df['3'])
    infant_df['arm_y'] = (infant_df[infant_elbow_y]-infant_df['4'])

    # mom_features = ['head_x', 'head_y', 'arm_x', 'arm_y']
    # infant_features = ['head_x', 'head_y', 'arm_x', 'arm_y']

    times = get_times(iid, age)
    if stage=='play':
        mom_stage = mom_df.iloc[times['start_play']:times['start_sf'],:]
        infant_stage = infant_df.iloc[times['start_play']:times['start_sf'],:]
    elif stage=='sf':
        mom_stage = mom_df.iloc[times['start_sf']:times['end_sf'],:]
        infant_stage = infant_df.iloc[times['start_sf']:times['end_sf'],:]
    else:
        mom_stage = mom_df.iloc[times['end_sf']:times['end_reunion'],:]
        infant_stage = infant_df.iloc[times['end_sf']:times['end_reunion'],:]
    mom_stage = mom_stage.loc[:,mom_features].values
    infant_stage = infant_stage.loc[:,infant_features].values
    # mom_stage = np.hstack((mom_stage.loc[:,mom_features], mom_stage.loc[:,mom_features].diff()))[1:,:]
    # infant_stage = np.hstack((infant_stage.loc[:,infant_features], infant_stage.loc[:,infant_features].diff()))[1:,:]

    X = mom_stage[:-1,:]
    Y = infant_stage[:-1,:]
    Z = mom_stage[1:,:]

    dmdc = DMDc(B=None)
    dmdc.fit(X,Y,Z)

    # return np.sum(np.abs(dmdc.A)), np.sum(np.abs(dmdc.B))
    return dmdc.A, dmdc.B
    # return 0, 0

def train_dmdc_distances_reverse(iid, age, mom_features, infant_features, stage):
    if stage not in ['play', 'sf', 'reunion']:
        raise Exception('stage must be one of [play, sf, reunion]')
    mom_df, infant_df = get_op(iid, age)
    mom_dist, infant_dist = get_normalization_distance(iid, age)
    mom_df = mom_df/mom_dist
    infant_df = infant_df/infant_dist

    times = get_times(iid, age)
    if stage=='play':
        mom_stage = mom_df.iloc[times['start_play']:times['start_sf'],:]
        infant_stage = infant_df.iloc[times['start_play']:times['start_sf'],:]
    elif stage=='sf':
        mom_stage = mom_df.iloc[times['start_sf']:times['end_sf'],:]
        infant_stage = infant_df.iloc[times['start_sf']:times['end_sf'],:]
    else:
        mom_stage = mom_df.iloc[times['end_sf']:times['end_reunion'],:]
        infant_stage = infant_df.iloc[times['end_sf']:times['end_reunion'],:]
    mom_stage = mom_stage.loc[:,mom_features].values
    infant_stage = infant_stage.loc[:,infant_features].values
    # mom_stage = np.hstack((mom_stage.loc[:,mom_features], mom_stage.loc[:,mom_features].diff()))[1:,:]
    # infant_stage = np.hstack((infant_stage.loc[:,infant_features], infant_stage.loc[:,infant_features].diff()))[1:,:]

    Y = mom_stage[:-1,:]
    X = infant_stage[:-1,:]
    Z = infant_stage[1:,:]


    dmdc = DMDc(B=None)
    dmdc.fit(X,Y,Z)

    # return np.sum(np.abs(dmdc.A)), np.sum(np.abs(dmdc.B))
    return dmdc.A, dmdc.B
    # return 0, 0

def get_var(iid, age, df, feature, stage):
    times = get_times(iid, age)
    if stage=='play':
        stage = df.iloc[times['start_play']:times['start_sf'],:]
    elif stage=='sf':
        stage = df.iloc[times['start_sf']:times['end_sf'],:]
    else:
        stage = df.iloc[times['end_sf']:times['end_reunion'],:]
    signal = stage[feature].values
    return np.var(signal)

def get_entropy(iid, age, df, feature, stage):
    times = get_times(iid, age)
    if stage=='play':
        stage = df.iloc[times['start_play']:times['start_sf'],:]
    elif stage=='sf':
        stage = df.iloc[times['start_sf']:times['end_sf'],:]
    else:
        stage = df.iloc[times['end_sf']:times['end_reunion'],:]
    signal = stage[feature].values
    signal = signal + np.abs(np.min(signal))
    return entropy(signal)

def get_vocalization_time(iid, age, stage):
    times = get_times(iid, age)
    df = pd.read_csv(f'mother-infant-interactions/F0_csv_infant/t{num_to_str(iid)}{num_to_str(age)}')
    if stage=='play':
        stage = df.iloc[times['start_play']:times['start_sf'],:]
    elif stage=='sf':
        stage = df.iloc[times['start_sf']:times['end_sf'],:]
    else:
        stage = df.iloc[times['end_sf']:times['end_reunion'],:]
    v = stage.baby.sum()/(stage.shape[0])
    return v

def get_stage(iid, age, stage, mom_features, infant_features):
    if stage not in ['play', 'sf', 'reunion']:
        raise Exception('stage must be one of [play, sf, reunion]')
    mom_df, infant_df = get_op(iid, age)
    mom_dist, infant_dist = get_normalization_distance(iid, age)
    mom_df = mom_df/mom_dist
    infant_df = infant_df/infant_dist

    times = get_times(iid, age)
    if stage=='play':
        mom_stage = mom_df.iloc[times['start_play']:times['start_sf'],:]
        infant_stage = infant_df.iloc[times['start_play']:times['start_sf'],:]
    elif stage=='sf':
        mom_stage = mom_df.iloc[times['start_sf']:times['end_sf'],:]
        infant_stage = infant_df.iloc[times['start_sf']:times['end_sf'],:]
    else:
        mom_stage = mom_df.iloc[times['end_sf']:times['end_reunion'],:]
        infant_stage = infant_df.iloc[times['end_sf']:times['end_reunion'],:]

    return mom_stage[mom_features].values, infant_stage[infant_features].values

def get_stage_unmasked(iid, age, stage, mom_features, infant_features):
    if stage not in ['play', 'sf', 'reunion']:
        raise Exception('stage must be one of [play, sf, reunion]')
    mom_df, infant_df = get_op_unmasked(iid, age)
    mom_dist, infant_dist = get_normalization_distance(iid, age)
    mom_df = mom_df/mom_dist
    infant_df = infant_df/infant_dist

    times = get_times(iid, age)
    if stage=='play':
        mom_stage = mom_df.iloc[times['start_play']:times['start_sf'],:]
        infant_stage = infant_df.iloc[times['start_play']:times['start_sf'],:]
    elif stage=='sf':
        mom_stage = mom_df.iloc[times['start_sf']:times['end_sf'],:]
        infant_stage = infant_df.iloc[times['start_sf']:times['end_sf'],:]
    else:
        mom_stage = mom_df.iloc[times['end_sf']:times['end_reunion'],:]
        infant_stage = infant_df.iloc[times['end_sf']:times['end_reunion'],:]

    return mom_stage[mom_features].values, infant_stage[infant_features].values

def windowed_dmdc_stage(iid, age, mom_features, infant_features, stage, sec, control, a):
    '''Output a dataframe with A and B at the {seconds}-long window for the given parameters
       control: 'mom' or 'infant'
    '''
    # get the mother and infant dataframes at the requested stage
    mom_stage, infant_stage = get_stage(iid, age, stage, mom_features, infant_features)
    mom_arm, infant_arm = get_arm_features_unmasked(iid, age)
    mom_stage_confidence, infant_stage_confidence = get_stage_unmasked(iid, age, stage, [str(3*mom_arm[1]+2)], [str(3*infant_arm[1]+2)])
    # print(str(3*mom_arm[1]+2))
    # determine who is the control
    if control=='mom':
        Y = mom_stage[:-1,:]; X = infant_stage[:-1,:]; Z = infant_stage[1:,:]
    else: 
        Y = infant_stage[:-1,:]; X = mom_stage[:-1,:]; Z = mom_stage[1:,:]
    seconds = np.min([sec, int(Y.shape[0]/FPS)])
    # for every __ seconds, get A and B for the interval
    i=0
    df = pd.DataFrame()
    while i + FPS*seconds < Y.shape[0]:
        j = int(i + FPS*seconds)
        Yi, Xi, Zi = Y[i:j], X[i:j], Z[i:j]
        dmdc = DMDc(B=None)
        try:
            dmdc.fit(Xi,Yi,Zi)
            A, B = dmdc.A, dmdc.B
            mom_arm_confidence, infant_arm_confidence = np.mean(mom_stage_confidence[i:j]>0), np.mean(infant_stage_confidence[i:j]>0)
            # print(mom_stage_confidence[i:j])
            window = pd.DataFrame(np.array([json.dumps(A.tolist()), json.dumps(B.tolist()), mom_arm_confidence, infant_arm_confidence]).reshape([1,-1]))
        except:
            window = pd.DataFrame(np.array([0, 0, 0, 0]).reshape([1,-1]))
        df = df.append(window)
        i=j
    df.columns = ['A', 'B', 'mom_c', 'infant_c']
    return df.reset_index(drop=True)

def windowed_dmdc_full(iid, age, seconds):
    ''' Create a single dataframe with A and B for each {seconds}-long interval
        at each stage for each control and feature set
    '''
    df = pd.DataFrame()
    cols = ['iid', 'age', 'features', 'stage', 'control', 'A', 'B']
    for features in ['arm', 'head', 'both']:
        if features == 'both':
            fs = ['head_x', 'head_y', 'arm_x', 'arm_y']
        else:
            fs = [f'{features}_x', f'{features}_y']
        for stage in ['play', 'sf', 'reunion']:
            for control in ['mom', 'infant']:
                results = windowed_dmdc_stage(iid, age, fs, fs, stage, seconds, control)
                results['iid'] = iid
                results['age'] = age
                results['features'] = features
                results['stage'] = stage
                results['control'] = control
                df = df.append(results, ignore_index=True)
    return df

def windowed_dmdc_full_audio(iid, age, seconds, a):
    ''' Create a single dataframe with A and B for each {seconds}-long interval
        at each stage for each control and feature set
    '''
    df = pd.DataFrame()
    cols = ['iid', 'age', 'features', 'stage', 'control', 'A', 'B']
    for features in ['arm', 'head', 'arm_head', 'audio', 'arm_audio', 'head_audio',  'all']:
        if features == 'all':
            fs = ['head_x', 'head_y', 'arm_x', 'arm_y']
            w_function = windowed_dmdc_stage_audio
        elif features == 'arm_audio':
            fs = ['arm_x', 'arm_y']
            w_function = windowed_dmdc_stage_audio
        elif features == 'head_audio':
            fs = ['head_x', 'head_y']
            w_function = windowed_dmdc_stage_audio
        elif features == 'audio':
            fs = []
            w_function = windowed_dmdc_stage_audio
        elif features == 'arm_head':
            fs = ['head_x', 'head_y', 'arm_x', 'arm_y']
            w_function = windowed_dmdc_stage
        elif features == 'arm':
            fs = ['arm_x', 'arm_y']
            w_function = windowed_dmdc_stage
        elif features == 'head':
            fs = ['head_x', 'head_y']
            w_function = windowed_dmdc_stage
            
        for stage in ['play', 'sf', 'reunion']:
            for control in ['mom', 'infant']:
                results = w_function(iid, age, fs, fs, stage, seconds, control, a)
                results['iid'] = iid
                results['age'] = age
                results['features'] = features
                results['stage'] = stage
                results['control'] = control
                df = df.append(results, ignore_index=True)
    return df

def windowed_dmdc_stage_audio(iid, age, mom_features, infant_features, stage, sec, control, a):
    '''Output a dataframe with A and B at the {seconds}-long window for the given parameters
       control: 'mom' or 'infant'
    '''
    # get the mother and infant dataframes at the requested stage
    mom_stage, infant_stage = get_stage(iid, age, stage, mom_features, infant_features)
    mom_arm, infant_arm = get_arm_features_unmasked(iid, age)
    mom_stage_confidence, infant_stage_confidence = get_stage_unmasked(iid, age, stage, [str(3*mom_arm[1]+2)], [str(3*infant_arm[1]+2)])
    mom_df, infant_df = get_f0(iid, age)
    mom_df = fill_gaps(mom_df, 'mom')
    infant_df = fill_gaps(infant_df, 'baby')

    times = get_times(iid, age)
    if stage=='play':
        mom_df = mom_df.iloc[times['start_play']:times['start_sf'],:].astype(float)
        infant_df = infant_df.iloc[times['start_play']:times['start_sf'],:].astype(float)
    elif stage=='sf':
        mom_df = mom_df.iloc[times['start_sf']:times['end_sf'],:].astype(float)
        infant_df = infant_df.iloc[times['start_sf']:times['end_sf'],:].astype(float)
    else:
        mom_df = mom_df.iloc[times['end_sf']:times['end_reunion'],:].astype(float)
        infant_df = infant_df.iloc[times['end_sf']:times['end_reunion'],:].astype(float)    

    mom_df.loc[mom_df.mom==0, 'pitch']=0
    infant_df.loc[infant_df.baby==0, 'pitch']=0
    mom_pitch = mom_df[['pitch']]
    infant_pitch = infant_df[['pitch']]
    if a == 'mom' or a == 'both':
        mom_stage = np.concatenate([mom_stage, mom_pitch.values], axis=1)
    if a == 'infant' or a == 'both':
        infant_stage = np.concatenate([infant_stage, infant_pitch.values], axis=1)
    if control=='mom':
        Y = mom_stage[:-1,:]; X = infant_stage[:-1,:]; Z = infant_stage[1:,:]
    else: 
        Y = infant_stage[:-1,:]; X = mom_stage[:-1,:]; Z = mom_stage[1:,:]
    seconds = np.min([sec, int(Y.shape[0]/FPS)])
    # for every __ seconds, get A and B for the interval
    i=0
    df = pd.DataFrame()
    while i + FPS*seconds < Y.shape[0]:
        j = int(i + FPS*seconds)
        Yi, Xi, Zi = Y[i:j], X[i:j], Z[i:j]
        dmdc = DMDc(B=None)
        try:
            dmdc.fit(Xi,Yi,Zi)
            A, B = dmdc.A, dmdc.B
            mom_arm_confidence, infant_arm_confidence = np.mean(mom_stage_confidence[i:j]>0), np.mean(infant_stage_confidence[i:j]>0)
            # print(mom_stage_confidence[i:j])
            window = pd.DataFrame(np.array([json.dumps(A.tolist()), json.dumps(B.tolist()), mom_arm_confidence, infant_arm_confidence]).reshape([1,-1]))
            print('==============================================')
            print(str(infant_pitch.values[i:j].tolist()))
            print('+++++++++++++++++++++++++++++++++++++++++')
            print(str(mom_pitch.values[i:j].tolist()))
            print('=============================================')
        except Exception as ex:
            window = pd.DataFrame(np.array([0, 0, 0, 0]).reshape([1,-1]))
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print('Didnt work:', str(infant_pitch.values[i:j].tolist()))
            # print(message)
        df = df.append(window)
        i=j
    df.columns = ['A', 'B', 'mom_c', 'infant_c']
    return df.reset_index(drop=True)


def train_dmdc_audio(iid, age, stage):
    if stage not in ['play', 'sf', 'reunion']:
        raise Exception('stage must be one of [play, sf, reunion]')
    mom_df, infant_df = get_f0(iid, age)
    times = get_times(iid, age)
    if stage=='play':
        mom_stage = mom_df.iloc[times['start_play']:times['start_sf'],:]
        infant_stage = infant_df.iloc[times['start_play']:times['start_sf'],:]
    elif stage=='sf':
        mom_stage = mom_df.iloc[times['start_sf']:times['end_sf'],:]
        infant_stage = infant_df.iloc[times['start_sf']:times['end_sf'],:]
    else:
        mom_stage = mom_df.iloc[times['end_sf']:times['end_reunion'],:]
        infant_stage = infant_df.iloc[times['end_sf']:times['end_reunion'],:]
    mom_stage = mom_stage[['mom']].values
    infant_stage = infant_stage[['baby']].values



def voc_amount(folder):
    data = pd.DataFrame()
    for txt in os.listdir(folder):
        try:
            age = int(txt[-8:-6])
            infant = int(txt[-12:-9])
            times = get_times(infant, age)
            start, end = times['start_sf']/FPS, times['end_sf']/FPS
            df = pd.read_csv(f'{folder}/{txt}')
            df.columns=['time', 'pitch']
            df.mask(df=='--undefined--', inplace=True)
            df=df.astype(float)
            sf = df.loc[(df.time>start)&(df.time<end)]
            percent = (~sf.pitch.isna()).mean()
            row = pd.DataFrame(np.array([infant, age, percent]).reshape([1,-1]))
            data = data.append(row, ignore_index=True)
        except:
            continue
    data.columns=['id', 'age', 'percent_sf_vocalization']
    return data

# voc_amount('C:/Users/klein/Desktop/wav_backup/F0').to_csv('sf_vocalization.csv', index=False)

 