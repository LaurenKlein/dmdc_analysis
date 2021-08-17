# openpose command:
# bin\OpenPoseDemo.exe --write_json C:\Users\klein\Desktop\imi_2\realtime_test\json_test2 --write_video C:\Users\klein\Desktop\imi_2\realtime_test\video.avi --display 0 --number_people_max 2
import os
import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from online_dmd.online_dmd import control
import json
import time
import shutil

both = True

def get_latest_poses(folder, i):
    # read in the last file of the folder
    last_file = os.listdir(folder)[i]
    return f'{folder}/{last_file}'

def get_keypoints(file):
    '''
    Get the set of left and right keypoints
    '''
    try:
        with open(file) as f:
            data = json.load(f)
        if len(data['people'])<2:
            left_keypoints, right_keypoints = None, None
        else:
            kp0 = data['people'][0]['pose_keypoints_2d']
            kp1 = data['people'][1]['pose_keypoints_2d']
            if kp0[0]<kp1[0]:
                left_keypoints, right_keypoints = kp0, kp1
            else:
                left_keypoints, right_keypoints = kp1, kp0
        return left_keypoints, right_keypoints
    except: 
        return None, None

def keypoints_to_row(left_keypoints, right_keypoints):
    left_row = pd.DataFrame(np.array(left_keypoints).reshape([1,-1]))
    right_row = pd.DataFrame(np.array(right_keypoints).reshape([1,-1]))
    return left_row, right_row

def update(left_person, right_person, folder, i):
    file = get_latest_poses(folder, i)
    left_keypoints, right_keypoints = get_keypoints(file)
    if left_keypoints is None:
        both=False
        left_row, right_row = left_person.iloc[-1,:] + 1, right_person.iloc[-1,:] +1
    else:
        both=True
        left_row, right_row = keypoints_to_row(left_keypoints, right_keypoints)
    left_person, right_person = left_person.append(left_row).reset_index(drop=True), right_person.append(right_row).reset_index(drop=True)
    if left_person.shape[0]>1000:
        return left_person.iloc[1:], right_person.iloc[1:]
    return left_person, right_person, both

def new_person():
    # return pd.DataFrame(np.zeros([1,75]))
    return pd.DataFrame(np.random.rand(100, 75))

def get_past_30(arr):
    if len(arr)<5: return arr
    return arr[-5:]

def rolling_av(arr, val):
    return (np.mean(get_past_30(arr))*5+val)/6

#

# start person track
folder = '/Users/klein/Desktop/imi_2/realtime_test/json_test2'
if os.path.isdir(folder):
    shutil.rmtree(folder)
    os.mkdir(folder)

# Instantiate new people
left_person, right_person = new_person(), new_person()
R_prev = 1
R2_prev = 1
r, r2 = [R_prev], [R2_prev]
l_prev, l2_prev = 1, 1
l, l2 = [l_prev], [l2_prev]


# # start figure
# fig = plt.figure()
# rects = plt.bar([1,2], [r[0],r2[0]])        # so that we can update data later
# plt.title('Relative Pose Influence')
# plt.xticks([1,2], ['Pose Coordination (Mom)', 'Pose Coordination (Infant)'])
# plt.ylabel('Relative Pose Influence')
# plt.ylim([0, 1.5])

# set second figure
fig2, (ax1, ax2) = plt.subplots(2, 1, sharey=True)
plt.rcParams["font.size"] = "12"
fig2.set_figheight(7)
fig2.set_figwidth(10)
line2, = ax1.plot([0,0],[0,0])
line1, = ax1.plot([0,0],[0,0])
line4, = ax2.plot([0,0],[0,0])
line3, = ax2.plot([0,0],[0,0]) 
ax2.set_xlabel('Time (s)')
plt.xlabel('Time', fontsize=12)


ax1.set_title('Relative Pose Influence')
ax1.set_ylabel('Relative Pose Influence', fontsize=12)
ax1.set_ylim([-1, 4])
ax1.set_xlim([0, 10])
# ax1.legend(['Pose Coordination (Mom)', 'Pose Coordination (Infant)'])
ax2.set_title('Movement Characteristic')
ax2.set_ylabel('Movement Characteristic', fontsize=12)
ax2.set_ylim([-1, 4])
ax2.set_xlim([0, 10])


# Select video to capture
vid_path = "/Users/klein/Desktop/imi_2/realtime_test/video.avi"
cap = cv2.VideoCapture(vid_path)
i=0

# Get current time
start = time.time()
t = time.time()
while True:
    try:
        if i%100==0:
            time.sleep(0.2)
        # grab the metrics #######################################
        # last_file = get_latest_poses(folder, i)
        # while last_file==prev_file:
        #         last_file = get_latest_poses(folder)
        both=True
        success=False
        while not success:
            try:
                left_person, right_person, both = update(left_person, right_person, folder, i)
                success=True
            except:
                success=False
        L, R = left_person.values, right_person.values
        ret, frame = cap.read()
        i=i+1
        if L.shape[0]>1:
            Z, X, Y = R[-91:-1,[0,1]] - R[-91:-1,[3,4]], R[-100:-10,[0,1]] - R[-100:-10,[3,4]], L[-100:-10,[0,1]]-L[-100:-10,[3,4]]
            Z2, X2, Y2 = L[-91:-1,[0,1]] - L[-91:-1,[3,4]], L[-100:-10,[0,1]] - L[-100:-10,[3,4]], R[-100:-10,[0,1]]-R[-100:-10,[3,4]]
            dmdc = control.DMDc(B=None)
            dmdc2 = control.DMDc(B=None)
            try:
                dmdc.fit(X, Y, Z)
                dmdc2.fit(X2, Y2, Z2)
                A, B = dmdc.A, dmdc.B
                A2, B2 = dmdc2.A, dmdc2.B
                wa, va = np.linalg.eig(A)
                wa2, va2 = np.linalg.eig(A2)
                w = np.max(np.abs(wa))
                w2 = np.max(np.abs(wa2))
                
                r.append(rolling_av(r, np.linalg.norm(B)/np.linalg.norm(A)))
                r2.append(rolling_av(r2, np.linalg.norm(B2)/np.linalg.norm(A2)))
                R_prev = r[-1]
                R2_prev = r2[-1]
                l.append(rolling_av(l, w))
                l2.append(rolling_av(l2, w2))
                l_prev = l[-1]
                l2_prev = l2[-1]

            except:
                print('Error')
                r.append(rolling_av(r, R_prev))
                r2.append(rolling_av(r2, R2_prev))
                l.append(rolling_av(l, l_prev))
                l2.append(rolling_av(l2, l2_prev))


        # show video frame, with extra info
        # if both participants are in the frame, draw green rectangle
        # faces = (L[-1,0]>0 and R[-1,0]>0)
        try:
            if i%2==0:
                font = cv2.FONT_HERSHEY_SIMPLEX
                if both: color=[100,200,100]; text = " "; t = time.time()
                else: color = [50,50,200]; text = "Please adjust camera or sitting position."
                time_text = "Time Remaining:"
                # cv2.rectangle(frame, (2,2), (frame.shape[1]-1, frame.shape[0]-1), color, thickness=5)
                cv2.rectangle(frame, (0,frame.shape[0]-30), (frame.shape[1], frame.shape[0]), [250, 250, 250])
                # cv2.rectangle(frame, (10,frame.shape[0]-10), (frame.shape[1]-10, frame.shape[0]-20), [100,200,100], thickness=2)
                cv2.rectangle(frame, (frame.shape[1],frame.shape[0]-30), (10+int(t-start), frame.shape[0]), color, cv2.FILLED)
                cv2.putText(frame, text, (10, frame.shape[0]-70), font, 0.8, color, thickness=2)
                cv2.putText(frame, time_text, (10, frame.shape[0]-40), font, 0.8, color, thickness=2)
                cv2.imshow("Parent Feedback", frame)
                k = cv2.waitKey(1) & 0xFF
                if k == 27:
                    break
            if t-start>600: t=20
        except:
            continue
    # update data for drawing #######################################
        # rects[0].set_height(r[-1])
        # rects[1].set_height(r2[-1])
        # fig.canvas.draw()
        # img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8,
        #         sep='')
        # img  = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        # img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        # cv2.imshow("plot",img)

        line1.set_xdata(np.linspace(0, len(r)/30, len(r)))
        line1.set_ydata(np.log(1+np.array(r)))
        line2.set_xdata(np.linspace(0, len(r2)/30, len(r2)))
        line2.set_ydata(np.log(1+np.array(r2)))

        line3.set_xdata(np.linspace(0, len(l)/30, len(l)))
        line3.set_ydata(np.array(l))
        line4.set_xdata(np.linspace(0, len(l2)/30, len(l2)))
        line4.set_ydata(np.array(l2))
        if len(r)>10*30:
            ax1.set_xlim([len(r)/30-10, len(r)/30])
            ax2.set_xlim([len(r)/30-10, len(r)/30])
        fig2.canvas.draw()
        img2 = np.fromstring(fig2.canvas.tostring_rgb(), dtype=np.uint8,
                sep='')
        img2  = img2.reshape(fig2.canvas.get_width_height()[::-1] + (3,))
        img2 = cv2.cvtColor(img2,cv2.COLOR_RGB2BGR)
        

        cv2.imshow("plot",img2)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
    except:
        continue
    
