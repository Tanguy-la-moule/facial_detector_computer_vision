from image_manipulation import load_frame, load_frames
from image_manipulation import make_video_from_frames, create_frames_from_video, save_pictures
from frame_treatment import *


def pass_frames_through_pipeline(name, frame_rate, frame_max, pipeline_path='./video/', image_path='./img/'):
    frames = []
    frame_number = 0
    other_frame_number = frame_rate
    # Absdiff treatment
    while frame_number <= frame_max:
        frame = load_frame(frame_number, name, image_path, grayscale=True)
        other_frame = load_frame(other_frame_number, name, image_path, grayscale=True)
        frames.append(apply_cv2(frame, other_frame))
        frame_number += frame_rate
        other_frame_number = frame_number - frame_rate
    video_name = name + '_1_absdiff'
    make_video_from_frames(frames, video_name, grey=True, video_path=pipeline_path)
    print('Done step 1 - absdiff ')

    # threshold treatment
    thresholded_frames = []
    for threshold in [20]: # 25
        for frame in frames:
            thresholded_frames.append(apply_threshold(frame, threshold))
        video_name = name + '_2_absdiff_threshold_' + str(threshold)
        make_video_from_frames(thresholded_frames, video_name, grey=True, video_path=pipeline_path)
        print('Done step 2 - threshold ' + str(threshold))

    # median filtering
    median_filtered_frames = []
    for conv_size in [5]:  # 3
        for frame in thresholded_frames:
            median_filtered_frames.append(apply_median_filter(frame, conv_size))
        video_name = name + '_3_absdiff_threshold_20_median_' + str(conv_size)
        make_video_from_frames(median_filtered_frames, video_name, grey=True, video_path=pipeline_path)
        print('Done step 3 - median filter ' + str(conv_size))

    # operations on masks (scipy)
    masks_results = []
    for closing_size in [20]:  # 10, 30
        for frame in median_filtered_frames:
            mask = convert_grey_image_to_mask(frame)
            masks_results.append(apply_fill_binary_holes(mask, closing_size))
        video_name = name + '_4_absdiff_threshold_20_median_5_mask_' + str(closing_size)
        make_video_from_frames(masks_results, video_name, grey=True, video_path=pipeline_path)
        print('Done step 4.1 - masks operations ' + str(closing_size))

    # creating bounding boxes
    frame_number = 0
    bbs_groups_by_frame = []
    original_images = load_frames(name, frame_rate, frame_max, image_path=image_path, grayscale=False)
    for min_area in [500]:
        while frame_number < len(original_images):
            contours = get_contours(masks_results[frame_number])
            bbs_groups_by_frame.append(get_bounding_boxes(contours, min_area, frame_number))
            frame_number += 1
    print('Done step 4.2 - created bounding boxes ' + str(closing_size))

    # creating videos with boxes on
    boxed_masks = []
    boxed_images = []
    for i, bounding_boxes in enumerate(bbs_groups_by_frame):
        boxed_masks.append(apply_boxes_to_image(masks_results[i], bounding_boxes, modified=False, grey=True))
        boxed_images.append(apply_boxes_to_image(original_images[i], bounding_boxes, modified=False, grey=False))

    video_name = name + '_5_absdiff_threshold_20_median_5_mask_20_contouring' + str(min_area)
    make_video_from_frames(boxed_masks, video_name, grey=False, video_path=pipeline_path)
    print('Done step 5 - Displaying bounding boxes on masks')

    video_name = name + '_6_absdiff_threshold_20_median_5_mask_20_contouring_100_color'
    make_video_from_frames(boxed_images, video_name, grey=False, video_path=pipeline_path)
    print('Done step 6 - rendering')

    # extracting pictures from biggest bounding boxes
    for min_area in [20000]:
        best_bbs = retrieve_best_bounding_boxes(bbs_groups_by_frame, min_area)
        pictures = get_pictures_from_bounding_boxes(best_bbs, original_images)
        save_pictures(pictures, name + '_7_area_' + str(min_area) + '_', image_path=pipeline_path)
    print('Done step 7 - extracting faces from bounding boxes - saved ' + str(len(pictures)) + ' pictures')

    modified_bbs = []
    modified_boxed_images = []
    for i, bounding_boxes in enumerate(bbs_groups_by_frame):
        modified_bbs.append(treat_bounding_box(bounding_boxes, 100))
        modified_boxed_images.append(apply_boxes_to_image(original_images[i], modified_bbs[-1], modified=True, grey=False))

    # Boxes on original video (modified)
    video_name = name + '_8_absdiff_threshold_20_median_5_mask_20_contouring_100_color_modified'
    make_video_from_frames(modified_boxed_images, video_name, grey=False, video_path=pipeline_path)
    print('Done step 8 - rendering and modified')

    # extracting pictures from corrected bounding boxes
    for min_area in [20000]:
        best_bbs = retrieve_best_bounding_boxes(modified_bbs, min_area)
        pictures = get_pictures_from_bounding_boxes(best_bbs, original_images)
        save_pictures(pictures, name + '_9_area_' + str(min_area) + '_', image_path=pipeline_path)
    print('Done step 9 - extracting faces from corrected bounding boxes - saved ' + str(len(pictures)) + ' pictures')
