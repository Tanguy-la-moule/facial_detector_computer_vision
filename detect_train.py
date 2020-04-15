from imutils import paths
import numpy as np
import imutils
import pickle
import cv2
import os

from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle

def detect_faces(dataset_path='./train_set',detector_path='./face_detection_model',confidence_lvl = 0.5,
                 embedding_model_path='./embeddings/openface_nn4.small2.v1.t7',
                 embedder_output = './embeddings/embedding.pickle'):
    images_paths = list(paths.list_images(dataset_path))
    proto_path = os.path.sep.join([detector_path, "deploy.prototxt"])
    model_path = os.path.sep.join([detector_path,"res10_300x300_ssd_iter_140000.caffemodel"])
    detector = cv2.dnn.readNetFromCaffe(proto_path, model_path)
    embedder = cv2.dnn.readNetFromTorch(embedding_model_path)

    knownEmbeddings = []
    knownNames = []

    cpt = 0
    n = len(images_paths)

    for image_path in images_paths:
        cpt+=1
        name = image_path.split(os.path.sep)[-2]

        image = cv2.imread(image_path)
        image = imutils.resize(image, width=600)
        (h, w) = image.shape[:2]
        imageBlob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0), swapRB=False, crop=False)

        detector.setInput(imageBlob)
        detections = detector.forward()

        if len(detections) > 0: # checking if there is at least one face
            i = np.argmax(detections[0, 0, :, 2])
            confidence = detections[0, 0, i, 2]
            if confidence > confidence_lvl: # confidence level to be pretty sure it is a face
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                face = image[startY:endY, startX:endX]
                faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                                                 (96, 96), (0, 0, 0), swapRB=True, crop=False)
                embedder.setInput(faceBlob)
                vec = embedder.forward()

                knownNames.append(name)
                knownEmbeddings.append(vec.flatten())

        print(str(cpt) + ' photos analyzed on ' + str(n))

    data = {"embeddings": knownEmbeddings, "names": knownNames}
    f = open(embedder_output, "wb")
    f.write(pickle.dumps(data))
    f.close()
    print('Embeddings preprocessed')

    return data

def training_model(data, recognizer_output = './embeddings/recognizer.pickle',le_output = './embeddings/le.pickle'):
    le = LabelEncoder()
    labels = le.fit_transform(data["names"])
    recognizer = SVC(C=1.0, kernel="linear", probability=True)
    recognizer.fit(data["embeddings"], labels)

    f = open(recognizer_output, "wb")
    f.write(pickle.dumps(recognizer))
    f.close()

    f = open(le_output, "wb")
    f.write(pickle.dumps(le))
    f.close()

    print('Model trained with the new dataset')

    return recognizer_output, le_output

if __name__ == "main":
    data = detect_faces()
    training_model(data)