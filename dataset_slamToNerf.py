import os
import sys
import shutil

import numpy as np
import cv2
from scipy.spatial.transform import Rotation
import scipy.misc

rgb_path="2022-12-21-15-57-16-car/rgb"
depth_path="2022-12-21-15-57-16-car/depth"

FrameTrajs=np.loadtxt("2022-12-21-15-57-16-car/CameraTrajectory.txt")#
KeyFrameTrajs=np.array(FrameTrajs[0:3000:30][:])#

KeyFrameTrajs_Rotation=KeyFrameTrajs[:,4:8]
Timestamp=KeyFrameTrajs[:,0]
nums=np.size(KeyFrameTrajs_Rotation,axis=0)
KeyFrameTrajs_Matrix=np.zeros([nums,4,4])

for i in range(0,nums):
    Rm=Rotation.from_quat(KeyFrameTrajs_Rotation[i,:])
    KeyFrameTrajs_Matrix[i,0:3,0:3]=Rm.as_matrix()
    KeyFrameTrajs_Matrix[i,0:3,3]=KeyFrameTrajs[i,1:4]
    KeyFrameTrajs_Matrix[i,3,:]=[0,0,0,1]

for i in range(0,nums):
    shutil.copyfile(os.path.join(rgb_path,'%.6f' % Timestamp[i]+".png"),os.path.join("output2/color",str(i)+".png"))
    shutil.copyfile(os.path.join(depth_path,'%.6f' % Timestamp[i]+".png"),os.path.join("output2/depth",str(i)+".png"))
    np.savetxt(os.path.join("output2/pose",str(i)+".txt"),KeyFrameTrajs_Matrix[i])
    
K_1=np.array([[384.2230835 ,   0.        , 315.03936768,   0.        ],
       [  0.        , 383.96340942, 245.00970459,   0.        ],
       [  0.        ,   0.        ,   1.        ,   0.        ],
       [  0.        ,   0.        ,   0.        ,   1.        ]])

np.savetxt("output2/intrinsic/intrinsic_color.txt",K_1)

depth_path="output2/depth"
for i in range(0,nums):
    img=cv2.imread(os.path.join(depth_path,str(i)+".png"),-1)
    img[img>5000]=0
    img[img==0]=0
    img[img>5]=255
    img_out=img.astype(np.uint8)
    cv2.imwrite(os.path.join("output2/mask",str(i)+".png"),img_out)