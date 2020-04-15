import numpy as np
import imutils
import pickle
import cv2
import os


def test_image(input_image,detector_path='./face_detection_model',
			   embedding_model_path='./embeddings/openface_nn4.small2.v1.t7',
			   recognizer_path='./embeddings/recognizer.pickle', le_path='./embeddings/le.pickle',
			   confidence_lvl=0.5):
	proto_path = os.path.sep.join([detector_path, "deploy.prototxt"])
	model_path = os.path.sep.join([detector_path, "res10_300x300_ssd_iter_140000.caffemodel"])
	detector = cv2.dnn.readNetFromCaffe(proto_path, model_path)

	embedder = cv2.dnn.readNetFromTorch(embedding_model_path)


	recognizer = pickle.loads(open(recognizer_path, "rb").read())
	le = pickle.loads(open(le_path, "rb").read())

	image = imutils.resize(input_image, width=600)
	(h, w) = image.shape[:2]

	imageBlob = cv2.dnn.blobFromImage(
		cv2.resize(image, (300, 300)), 1.0, (300, 300),
		(104.0, 177.0, 123.0), swapRB=False, crop=False)

	detector.setInput(imageBlob)
	detections = detector.forward()

	for i in range(0, detections.shape[2]):
		confidence = detections[0, 0, i, 2]

		if confidence > confidence_lvl:
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			face = image[startY:endY, startX:endX]

			(fH, fW) = face.shape[:2]
			# ensure the face width and height are sufficiently large
			if fW < 20 or fH < 20:
				continue

			faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96),
				(0, 0, 0), swapRB=True, crop=False)
			embedder.setInput(faceBlob)
			vec = embedder.forward()

			preds = recognizer.predict_proba(vec)[0]
			j = np.argmax(preds)
			proba = preds[j]
			name = le.classes_[j]

			text = "{}: {:.2f}%".format(name, proba * 100)
			y = startY - 10 if startY - 10 > 10 else startY + 10
			cv2.rectangle(image, (startX, startY), (endX, endY),
				(0, 0, 255), 2)
			cv2.putText(image, text, (startX, y),
				cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

			cv2.imshow("Image", image)
			cv2.waitKey(0)

			return name, proba, preds

	# if no faces is recognized return unknown and proba of 0
	return 'unknown', 0, []

if __name__ == 'main':
	image = cv2.imread('./pipeline/ben_9_area_20000_0.jpg')
	test_image(image)
