# Import module
import os
from nudenet import NudeDetector
import argparse
import time
import cv2
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--file_path", type=str, help="path to input image/video")
parser.add_argument("-th", "--threshold_score", type=float, default=0.0,
                    help="minimum probability to filter weak detections")
args = parser.parse_args()

IMG_EXTS = [".jpg", ".png"]
VDO_EXTS = [".mp4"]
BATCH_SIZE = 8
BOOLEAN = 1
SAFE_CLASSES = ["FACE_F", "FACE_M", "COVERED_FEET", "EXPOSED_FEET"]

# initialize detector (downloads the checkpoint file automatically the first time)
detector = NudeDetector()  # detector = NudeDetector('base') for the "base" version of detector.

input_file = args.file_path
if os.path.splitext(input_file)[-1] in IMG_EXTS:
    res = detector.detect(input_file)
    print(res)
elif os.path.splitext(input_file)[-1] in VDO_EXTS:

    if os.path.exists(input_file + '.pickle'):
        with open(input_file + '.pickle', 'rb') as handle:
            detections = pickle.load(handle)
    else:
        detections = detector.detect_video(input_file, batch_size=BATCH_SIZE, show_progress=BOOLEAN)  # , mode='fast')
        with open(input_file + '.pickle', 'wb') as handle:
            pickle.dump(detections, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(detections)

    # initialize the video stream, allow the camera sensor to warmup,
    # and initialize the FPS counter
    print("[INFO] starting video stream...")
    vs = cv2.VideoCapture(input_file)
    time.sleep(2.0)
    # fps = FPS().start()
    frame_i = 1
    while True:
        grabbed, frame = vs.read()
        if not grabbed:
            break
        # frame = imutils.resize(frame, width=400)
        # (h, w) = frame.shape[:2]
        try:
            frame_detections = detections["preds"][frame_i]
            print(frame_detections)
            for res in frame_detections:
                box, score, label = res['box'], res['score'], res['label']
                x1, y1, x2, y2 = box
                if score > args.threshold_score and label not in SAFE_CLASSES:
                    frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), -1)
                    frame = cv2.rectangle(frame, (20, 20), (60, 60), (0, 255, 0), 2)
        except:
            pass
        cv2.imshow("Frame", frame)
        frame_i += 1
        # time.sleep(0.02)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    print("[INFO] approx. FPS: {:.2f}".format(vs.get(cv2.CAP_PROP_FPS)))
    vs.release()
    cv2.destroyAllWindows()

# fast mode is ~3x faster compared to default mode with slightly lower accuracy.
# detector.detect('path_to_image', mode='fast')
# Returns [{'box': LIST_OF_COORDINATES, 'score': PROBABILITY, 'label': LABEL}, ...]

# Detect video
# batch_size is optional; defaults to 2
# show_progress is optional; defaults to True
# detector.detect_video(args.video_path, batch_size=BATCH_SIZE, show_progress=BOOLEAN)
# fast mode is ~3x faster compared to default mode with slightly lower accuracy.
# detector.detect_video(args.video_path, batch_size=BATCH_SIZE, show_progress=BOOLEAN, mode='fast')
# Returns {"metadata": {"fps": FPS, "video_length": TOTAL_N_FRAMES, "video_path": 'path_to_video'},
#          "preds": {frame_i: {'box': LIST_OF_COORDINATES, 'score': PROBABILITY, 'label': LABEL}, ...], ....}}
