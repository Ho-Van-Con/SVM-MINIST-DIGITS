import matplotlib.pyplot as plt
import cv2
import numpy as np
from sklearn import datasets, svm
import glob

digits = datasets.load_digits()
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

# Create a classifier: a support vector classifier
classifier = svm.SVC(gamma=0.0001)
classifier.fit(data, digits.target)

# Read images from folders
path = glob.glob("data\*\*.png")
img = [] # Store images to show
data_img = [] # Store images to processing
for index in path:
    temp_img = cv2.imread(index) # Read image
    temp_img = cv2.cvtColor(temp_img,cv2.COLOR_BGR2GRAY) # Conver to Gray Image
    temp_data_img = cv2.resize(temp_img,(8,8)) //16 #Change 8-bit image to 4-bit image
    data_img.append(temp_data_img) # Add image to data.
    img.append(temp_img) # Add image to data.

n_samples_test = len(img)
data_img = np.array(data_img)
data_img = data_img.reshape(n_samples_test,-1)
preditions = classifier.predict(data_img)
images_and_predictions = list(zip(img, preditions))
for index, (image, prediction) in enumerate(images_and_predictions[:n_samples_test]):
    plt.subplot(n_samples_test//5 +1, 5, index+1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r)
    plt.title('Prediction: %i' % prediction)
plt.show()