from imutils import paths
import numpy as np
import cv2
def preprocess_img(img):
    # apply opencv preprocessing
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (640, 640))
    img = np.array(img, dtype=float)
    return img
def load_data():
    dataset ="./midterm"
    print("[INFO] veri yukleniyor...")
    imagePaths = list(paths.list_images(dataset))
    trainX = []
    trainY = []
    testX = []
    testY = []
    for imagePath in imagePaths:
        # Gets Image Label By name of folder
        label = imagePath.split('\\')[-2]
        # Gets Image Type
        img_type = imagePath.split('\\')[-3]
        #is it test or train data
        category = imagePath.split('\\')[-4]
        if img_type == 'RAW IMAGES':
            # Görüntüyü oku ve önişle
            img = cv2.imread(imagePath)
            image = preprocess_img(img)
            # update the data and labels lists, respectively
            if category == 'Test':
                testX.append(image)
                testY.append(label)
            elif category == 'Train':
                trainX.append(image)
                trainY.append(label)

    trainX = np.array(trainX, dtype=float)
    trainY = np.array(trainY)

    testX = np.array(testX, dtype=float)
    testY = np.array(trainY)
    return trainX, testX, trainY, testY
def main():
    # Veriyi yükle ve eğitim ve test olarak ayır.
    trainX, testX, trainY, testY = load_data()
    # Kodu yaz
if __name__ == '__main__':
    main() 