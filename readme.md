Facial detector to see if an authorized person is entering the room

Folders:
- train_set: contains photos for training the model to recognize who the people wanted
- embeddings: contains pickle files resulting from the training set
- face_detection_model: contains the model pre-trained used
- video: contains the videos used for the experiment

Scripts:
- detect_train.py: script that trains the pre-trained model with our specific training set
- facial_detection.py: script that recognizes a person on a picture
- frame_treatment.py: methods for frames of the video manipulation
- image_manipulation.py: methods for images manipulation
- main.py: main program of the video, train the model, then extract frames from the testing videos, and finally passes it through the pipeline
- pipeline.py: extract photos of faces from a video and try to recognize who is it
