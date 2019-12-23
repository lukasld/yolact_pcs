import torchvision
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
import os
import pickle

def run():
    fNames = [n for n in os.listdir('./resources') if 'png' in n]
    fNames = [os.path.join('./resources', n) for n in fNames]
    outNames = [n.replace('.png', '_det.pkl') for n in fNames]
    #load the torch model
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    
    for finp,fout in zip(fNames, outNames):
        im = plt.imread(finp)
        im = np.array(np.transpose(im, (1, 0, 2))[:, ::-1])
        im = cv2.resize(im, (480, 848))
        image_tensor = torch.from_numpy(im).permute(2, 0, 1).unsqueeze(0).float()
        dat =  model(image_tensor)
        with open(fout, 'wb') as fid:
            pickle.dump(dat, fid)

if __name__=="__main__":
    run()
