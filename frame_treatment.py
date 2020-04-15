import cv2
import imutils
from scipy.ndimage.morphology import binary_closing, binary_fill_holes
import numpy as np


def apply_cv2(frame, other_frame):
    treated_frame = cv2.absdiff(frame, other_frame)
    return treated_frame


def apply_threshold(frame, threshold):
    ret, treated_frame = cv2.threshold(frame, threshold, 255, cv2.THRESH_BINARY)
    return treated_frame


def apply_median_filter(frame, conv_size):
    treated_frame = cv2.medianBlur(frame, conv_size)
    return treated_frame


def convert_grey_image_to_mask(frame):
    frame[frame == 0] = 0
    frame[frame == 255] = 1
    return frame


def apply_fill_binary_holes(frame, closure_size):
    binary_closed_frame = binary_closing(frame, structure=np.ones((closure_size, closure_size)))
    result = binary_fill_holes(binary_closed_frame).astype(np.uint8())
    result[result == 0] = 0
    result[result == 1] = 255
    return result


def get_contours(mask):
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    return contours


def get_bounding_boxes(contours, min_area, frame_number):
    bounding_boxes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_area:
            (x, y, w, h) = cv2.boundingRect(contour)
            if (w >= 2 * h) or (h >= 2 * w):  # Checking for rectangle boxes
                bounding_boxes.append({'i': frame_number, 'x': x, 'y': y, 'w': w, 'h': h})
    return bounding_boxes


def treat_bounding_box(bbs, min_distance):
    found_match = True
    if len(bbs) <= 1:
        return bbs
    while found_match:
        found_match_this_turn = False
        for i in range(len(bbs) - 1):
            for j in range(i + 1, len(bbs)):
                if not found_match_this_turn:
                    distance_x = bbs[i]['x'] + bbs[i]['w']/2 - (bbs[j]['x'] + bbs[j]['w']/2)
                    distance_y = bbs[i]['y'] + bbs[i]['h']/2 - (bbs[j]['y'] + bbs[j]['h']/2)
                    diagonals_average = (np.sqrt(bbs[i]['w']**2 + bbs[i]['h']**2) + np.sqrt(bbs[j]['w']**2 + bbs[j]['h']**2))/2
                    if np.sqrt(distance_x**2 + distance_y**2) - diagonals_average < min_distance:
                        found_match_this_turn = True
                        new_bbs = []
                        if i > 0:
                            for index in range(i):
                                new_bbs.append(bbs[index])
                        if j - i > 1:
                            for index in range(i+1, j):
                                new_bbs.append(bbs[index])
                        if j < len(bbs):
                            for index in range(j+1, len(bbs)):
                                print('end' + str(len(range(j+1, len(bbs)))))
                                new_bbs.append(bbs[index])
                        new_i = bbs[i]['i']
                        new_x = min(bbs[i]['x'], bbs[j]['x'])
                        new_w = max(bbs[i]['x'] + bbs[i]['w'], bbs[j]['x'] + bbs[j]['w']) - new_x
                        new_y = min(bbs[i]['y'], bbs[j]['y'])
                        new_h = max(bbs[i]['y'] + bbs[i]['h'], bbs[j]['y'] + bbs[j]['h']) - new_y
                        new_bbs.append({'i': new_i, 'x': new_x, 'y': new_y, 'w': new_w, 'h': new_h})
                        bbs = new_bbs
        if not found_match_this_turn:
            found_match = False
            return bbs


def retrieve_best_bounding_boxes(bbs_by_frames, min_area, number=5):
    ordered_bbs = []
    for bbs in bbs_by_frames:
        for bb in bbs:
            print(bb)
            i = 0
            bb['a'] = bb['w'] * bb['h']
            while i < len(ordered_bbs) and bb['a'] < ordered_bbs[i]['a']:
                i += 1
            ordered_bbs.insert(i, bb)
    best_bbs = []
    for i in range(number):
        if ordered_bbs[i]['a'] > min_area:
            best_bbs.append(ordered_bbs[i])
    return best_bbs


def get_pictures_from_bounding_boxes(bbs, frames, hide_bottoms=True):
    result = []
    for bb in bbs:
        if hide_bottoms:
            result.append(frames[bb['i']][bb['y']:bb['y'] + int(bb['h']/2), bb['x']:bb['x'] + bb['w']])
        else:
            result.append(frames[bb['i']][bb['y']:bb['y']+bb['h'], bb['x']:bb['x']+bb['w']])
    return result


def apply_boxes_to_image(image, bounding_boxes, modified=False, grey=False):
    if grey:
        result = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        result = image[:][:][:]
    for box in bounding_boxes:
        if modified:
            result = cv2.rectangle(result, (box['x'], box['y']), (box['x'] + box['w'], box['y'] + box['h']), (0, 0, 255), 2)
        else:
            result = cv2.rectangle(result, (box['x'], box['y']), (box['x'] + box['w'], box['y'] + box['h']),
                                   (0, 255, 0), 2)
    return result
