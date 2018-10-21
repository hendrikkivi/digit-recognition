import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.misc.pilutil import imresize
from sklearn import datasets
from sklearn.externals import joblib
from sklearn.neighbors import KNeighborsClassifier
from skimage.feature import hog

IMAGE_FILE = 'test.png'
IMG_HEIGHT = 28
IMG_WIDTH = 28

def getDigits(contours, hierarchy):
    hierarchy = hierarchy[0]
    bounding_rectangles = [cv2.boundingRect(ctr) for ctr in contours]   
    final_bounding_rectangles = []
    u, indices = np.unique(hierarchy[:,-1], return_inverse=True)
    most_common_heirarchy = u[np.argmax(np.bincount(indices))]
    
    for r,hr in zip(bounding_rectangles, hierarchy):
        x,y,w,h = r
        if ((w*h)>250) and (10 <= w <= 200) and (10 <= h <= 200) and hr[3] == most_common_heirarchy: 
            final_bounding_rectangles.append(r)    

    return final_bounding_rectangles

def pixels_to_hog_20(features):
    list_hog_fd = []
    for feature in features:
        fd = hog(feature.reshape((28, 28)), orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
        list_hog_fd.append(fd)
    hog_features = np.array(list_hog_fd, 'float64')
    return np.float32(hog_features)

def getClassifier():
    dataset = datasets.fetch_mldata("MNIST Original")
    features = np.array(dataset.data, 'int16')
    labels = np.array(dataset.target, 'int')

    features = pixels_to_hog_20(features)

    classifier = KNeighborsClassifier(3)
    print("Training...")
    classifier.fit(features, labels)
    joblib.dump(classifier, "digitsKnnHog.pkl", compress=3)
    

    return classifier


def predictImage():
    #classifier = getClassifier()
    
    classifier = joblib.load("digitsKnnHog.pkl")
    
    image = cv2.imread(IMAGE_FILE)
    
    imgray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    imgray = cv2.GaussianBlur(imgray, (5, 5), 0)
    
    plt.imshow(imgray)
    kernel = np.ones((5,5),np.uint8)
    
    ret,thresh = cv2.threshold(imgray,127,255,0)
    thresh = cv2.erode(thresh,kernel,iterations = 1)
    thresh = cv2.dilate(thresh,kernel,iterations = 1)
    thresh = cv2.erode(thresh,kernel,iterations = 1)
    
    _,contours,hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    digits_rectangles = getDigits(contours,hierarchy)
    
    for rect in digits_rectangles:
        x,y,w,h = rect
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
        im_digit = imgray[y:y+h,x:x+w]    
        im_digit = (255-im_digit)    
        im_digit = imresize(im_digit,(IMG_WIDTH, IMG_HEIGHT))
        digit = pixels_to_hog_20([im_digit])  
        prediction = classifier.predict(digit)
        print(prediction)
        cv2.putText(image, str(int(prediction[0])), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    cv2.imwrite('result.png',image)
    cv2.imshow("Predictions", image)
    cv2.waitKey()
    
    
predictImage()