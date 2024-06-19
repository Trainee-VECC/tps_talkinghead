from torch import nn
import torch
from torchvision import models
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from sklearn.preprocessing import MinMaxScaler
import numpy as np


# Create an FaceLandmarker object.
base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
options = vision.FaceLandmarkerOptions(base_options=base_options,
                                       output_face_blendshapes=False,
                                       output_facial_transformation_matrixes=False,
                                       num_faces=10)
detector = vision.FaceLandmarker.create_from_options(options)

class KPDetector(nn.Module):
    """
    Predict K*5 keypoints.
    """

    def __init__(self, num_tps, **kwargs):
        super(KPDetector, self).__init__()
        self.num_tps = num_tps
        self.detector=detector
        self.fg_encoder = models.resnet18(pretrained=False)
        num_features = self.fg_encoder.fc.in_features
        self.fg_encoder.fc = nn.Linear(num_features,(self.num_tps)*5*2)
        self.indexes=[21,67,151,297,251,  #forehead
                      46,66,8,296,276,    #eyebrows
                     226,161,158,132,163, #right eye
                     446,388,385,362,390, #left eye
                     240,237,2,457,460,   #nose
                     61,39,13,269,291,    #upper lip
                     78,181,15,405,308,   #lower lip
                     172,150,152,378,397, #chin 
                     101,50,147,132,58,   # right cheek
                     330,280,376,361,288] #left cheek
        
    def forward(self, image):
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        fg_kp = self.fg_encoder(image) 
        print(fg_kp.get_device())
        frame=np.transpose(image.detach().cpu().numpy(), [0, 2, 3, 1])[0]*255
        frame=frame.astype(np.uint8)
        frame=mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        data=detector.detect(frame)
        if len(data.face_landmarks)>0:
            arr=[]
            for j in range(478):
                i=data.face_landmarks[0][j]
                arr.append([i.x,i.y])
            arr=np.array(arr,dtype=np.float16)
            arr=MinMaxScaler().fit_transform(arr)
            flag=True
        else:
            arr=np.zeros((478,2)) 
            flag=False
        arr=arr[self.indexes]
        mp_kp=torch.tensor(arr,dtype=torch.float,device=device)
        fg_kp = self.fg_encoder(image)  
        bs, _, = fg_kp.shape
        fg_kp=torch.cat([mp_kp,fg_kp.view((self.num_tps)*5,2)])
        fg_kp = torch.sigmoid(fg_kp)
        out = {'fg_kp': fg_kp.view(bs,(self.num_tps+10)*5, -1),'face_found':flag}
        return out