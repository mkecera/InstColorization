import cv2
import numpy as np
import pickle
import tqdm
import os


def preprocess(frame):
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return img_gray


def prepare_testing_data(video_path):
    frames = []
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm.tqdm(total=frame_count)

    i = 0
    while cap.isOpened():
        i += 1
        pbar.update(1)
        ret, frame = cap.read()
        if ret:
            preprocessed_frame = preprocess(frame)
            # frames.append(frame)
            cv2.imwrite('./data/testing/target/' + str(i) + '.jpg', frame)
            cv2.imwrite('./data/testing/input/' + str(i) + '.jpg', preprocessed_frame)
    cap.release()
    cv2.destroyAllWindows()
    return frames


def prepare_training_data(training_path):
    j = 0
    for filename in os.listdir(training_path):
        if filename.endswith(".mp4"):
            j += 1

            if os.path.isdir(training_path + str(j)) is False:
                print('Create path: {0}'.format(training_path + str(j) + '/target/'))
                print('Create path: {0}'.format(training_path + str(j) + '/input/'))
                os.makedirs(training_path + str(j) + '/target/')
                os.makedirs(training_path + str(j) + '/input/')


            frames = []
            cap = cv2.VideoCapture(video_path+filename)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            pbar = tqdm.tqdm(total=3000)

            i = 0
            while i < 3000:
                i += 1
                pbar.update(1)
                ret, frame = cap.read()
                if ret:
                    preprocessed_frame = preprocess(frame)
                    cv2.imwrite(training_path + str(j) + '/target/' + str(i) + '.jpg', frame)
                    cv2.imwrite(training_path + str(j) + '/input/' + str(i) + '.jpg', preprocessed_frame)
            cap.release()
            cv2.destroyAllWindows()

    return frames


if __name__ == '__main__':
    # video_path = './data/testing/input.mp4'
    # prepare_testing_data(video_path)
    video_path = './data/training/'
    prepare_training_data(video_path)
