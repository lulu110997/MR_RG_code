import sys
import cv2
import mediapipe as mp

FILE = '/home/louis/Videos/2dof_ee_OA.mp4'
PATH_TO_MODEL = '/home/louis/Github/MR_RG_code/hand_landmarker.task'
MILLI = 1000

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
# Create a hand landmarker instance with the video mode:
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=PATH_TO_MODEL),
    running_mode=VisionRunningMode.VIDEO)


def video_frames(video):
    """
    Generator function for extracting images from videos
    Args:
        video: string | path to the video to read
    Returns: np.ndarray | Frames from each video
    """
    cap = cv2.VideoCapture(video)
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret:
            yield frame
        else:
            cap.release()
            cv2.destroyAllWindows()
            raise Exception(f'Cannot read video frame from {video}')

def get_fps(video):
    """
    Returns the video frame rate
    Args:
        video: string | path to the video to read
    Returns: int | Frames rate for the video file
    """
    cap = cv2.VideoCapture(video)
    return round(cap.get(cv2.CAP_PROP_FPS), 0)

try:
    with HandLandmarker.create_from_options(options) as landmarker:
        # The landmarker is initialized. Use it here.
        # Load the frame rate of the video using OpenCV’s CV_CAP_PROP_FPS
        # The three Kinect V2 cameras are triggered to capture the assembly activities  simultaneously in real
        # time (∼24 fps)
        # There are in total 1113 RGB videos and 371 depth videos (top view). Overall, the dataset contains 3,046,977
        # frames (∼35.27h) of footage with an average of 2735.2 frames per video (∼1.89min).
        # The dataset contains a total of 16,764 annotated actions with an average of 150 frames per action (∼6sec)
        # TODO: For a full list of action names and ids, see supplemental
        # Temporally, we specify the boundaries (start and end frame) of all atomic actions in the video from a pre-defined set.
        # TODO: atomic actions?
        # We also annotated the human skeleton of the subjects involved assembly... Annotated 12 body joints... Due to
        # occlusion with furniture, self-occlusions and uncommon  human poses, we include a confidence value between 1
        # and 3 along with the annotation
        # The dataset contains 2D human joint annotations in the COCO format [39] for 1% of frames, the same keyframes
        # selected for instance segmentation, which cover a diverse range of human poses across each video.
        # We also obtain pseudo-ground-truth 3D annotations by fine-tuning a Mask R-CNN [22] 2D joint detector on the
        # labeled data, and triangulating the detections of the model from the three calibrated camera views
        # https://arxiv.org/pdf/2007.00394.pdf
        fps = get_fps(FILE)  #TODO: fps for ikea-asm always 30?
        ts = 0.0
        print(fps)
        # TODO: You’ll need it to calculate the timestamp for each frame.
        for i in video_frames(FILE):
            # brg2rgb conversion as mp expects rgb images
            im = cv2.cvtColor(i, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=im)

            # Perform hand landmarks detection on the provided single image
            hand_landmarker_result = landmarker.detect_for_video(mp_image, int(ts))
            ts += round(1.0/fps * MILLI, 0)


except Exception as e:
    cv2.destroyAllWindows()
    raise e
