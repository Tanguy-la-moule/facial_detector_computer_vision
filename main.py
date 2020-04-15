from image_manipulation import create_frames_from_video
from pipeline import pass_frames_through_pipeline
from detect_train import training_model, detect_faces


# train the agent with our dataset
training_model(detect_faces())

# creates photo frames from the video in the './img/<name>' directory
create_frames_from_video('ben')
create_frames_from_video('tanguy')

# passes frames created through the homemade 'face identification' pipeline
pass_frames_through_pipeline('ben', 1, 207, pipeline_path='./pipeline/', image_path="./img/ben")
pass_frames_through_pipeline('tanguy', 1, 447, pipeline_path='./pipeline/', image_path="./img/tanguy")
