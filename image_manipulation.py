from matplotlib import pyplot as plt
import cv2
import os


def make_video_from_frames(frames, name, grey=False, video_path='./video/'):
    height = len(frames[0])
    width = len(frames[0][0])
    out = cv2.VideoWriter(os.path.join(video_path, name + '.avi'), cv2.VideoWriter_fourcc(*'DIVX'), 24, (width, height))
    for frame in frames:
        if grey:
            colored_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            out.write(colored_frame)
        else:
            out.write(frame)
    out.release()


def create_frames_from_video(name, frame_rate=1, video_path='./video/', image_path='./img/', extension='.avi'):
    cap = cv2.VideoCapture(os.path.join(video_path, name + extension))
    frame_number = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_number % frame_rate == 0:
            print(frame_number)
            frame_name = name + '_' + str(frame_number) + '.jpg'
            cv2.imwrite(os.path.join(image_path + name, frame_name), frame)
        frame_number += 1
    cap.release()
    cv2.destroyAllWindows()


def load_frame(frame, name, image_path='./img/', grayscale=False):
    full_path = os.path.join(image_path, name + '_' + str(frame) + '.jpg')
    if grayscale:
        image = cv2.imread(full_path)
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        return cv2.imread(full_path, cv2.IMREAD_UNCHANGED)


def load_frames(name, frame_rate, frame_max, image_path='./img/', grayscale=False):
    frames = []
    frame_number = 0
    while frame_number <= frame_max:
        full_path = os.path.join(image_path, name + '_' + str(frame_number) + '.jpg')
        if grayscale:
            image = cv2.imread(full_path)
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            frames.append(cv2.imread(full_path, cv2.IMREAD_UNCHANGED))
        frame_number += frame_rate
    return frames


def show_image(frame, grey=False):
    if grey:
        plt.imshow(frame, cmap='gray')
    else:
        plt.imshow(frame)
    plt.show()


def save_pictures(pictures, name, image_path='./img/'):
    for index, picture in enumerate(pictures):
        frame_name = name + str(index) + '.jpg'
        cv2.imwrite(os.path.join(image_path, frame_name), picture)
