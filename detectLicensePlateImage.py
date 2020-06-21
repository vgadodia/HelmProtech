# remove warning message
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# required library
import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import detect_lp
from os.path import splitext, basename
from keras.models import model_from_json

def load_model(path):
    try:
        path = splitext(path)[0]
        with open('%s.json' % path, 'r') as json_file:
            model_json = json_file.read()
        model = model_from_json(model_json, custom_objects={})
        model.load_weights('%s.h5' % path)
        print("Loading model successfully...")
        return model
    except Exception as e:
        print(e)

wpod_net_path = "wpod-net.json"
wpod_net = load_model(wpod_net_path)

def preprocess_image(image_path,resize=False):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255
    if resize:
        img = cv2.resize(img, (224,224))
    return img

def get_plate(image_path):
    Dmax = 608
    Dmin = 288
    vehicle = preprocess_image(image_path)
    ratio = float(max(vehicle.shape[:2])) / min(vehicle.shape[:2])
    side = int(ratio * Dmin)
    bound_dim = min(side, Dmax)
    _ , LpImg, _, cor, final_x, final_y = detect_lp(wpod_net, vehicle, bound_dim, lp_threshold=0.5)
    return vehicle, LpImg, cor, final_x, final_y

def get_coordinates(image_path):
    vehicle, LpImg,cor, final_x, final_y = get_plate(image_path)
    return LpImg, final_x, final_y

"""print("Detect %i plate(s) in"%len(LpImg),splitext(basename(image_path))[0])

for i in range(len(LpImg)):
    
    cv2.imshow('Output', LpImg[i])
    cv2.waitKey()
    cv2.destroyAllWindows()"""

#plt.show()