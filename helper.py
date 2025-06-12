import numpy as np
import joblib
import cv2
import warnings
warnings.filterwarnings('ignore')

model = joblib.load('lr_model.joblib')
scaler = joblib.load('scaler.joblib')

def img_to_vec(img_path):
    img_arr = cv2.imread(img_path)
    mean_pixels = np.mean(img_arr,axis=-1).ravel()
    median_pixels = np.median(img_arr,axis=-1).ravel()
    max_pixels = np.max(img_arr,axis=-1).ravel()
    min_pixels = np.min(img_arr,axis=-1).ravel()
    std_pixels = np.std(img_arr,axis=-1).ravel()
    twenty_pixels = np.percentile(img_arr,25,axis=-1).ravel()
    fifty_pixels = np.percentile(img_arr,50,axis=-1).ravel()
    seventy_pixels = np.percentile(img_arr,75,axis=-1).ravel()
    full_img_arr = np.concatenate((mean_pixels,median_pixels,max_pixels,min_pixels,std_pixels,twenty_pixels,fifty_pixels,seventy_pixels),axis=-1)
    return np.array(full_img_arr)

def predict_class(vec):
    vec = scaler.transform(vec.reshape(1,-1))
    pred = model.predict(vec)
    prob = np.max(model.predict_proba(vec))
    return str(pred[0]), float(round(prob,3))

# img_path = '10009.png'
# img_vec = img_to_vec(img_path)
# pred,prob = predict_class(img_vec)
# print(pred,prob)