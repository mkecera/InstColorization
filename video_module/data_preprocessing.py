import cv2
import numpy as np
import pickle
import tqdm


def preprocess(frame):
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return img_gray


def save_to_pickle(frames, file_name):
    with open('./data/' + file_name + '.pickle', 'wb') as fp:
        pickle.dump(frames, fp)


def convert_movie_to_preprocessed_frames(video_path):
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
            cv2.imwrite('./data/target/' + str(i) + '.jpg', frame)
            cv2.imwrite('./data/input/' + str(i) + '.jpg', preprocessed_frame)
    cap.release()
    cv2.destroyAllWindows()
    return frames


if __name__ == '__main__':
    video_path = './data/input.mp4'
    convert_movie_to_preprocessed_frames(video_path)
