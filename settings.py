from pathlib import Path
import sys

# Get the absolute path of the current file
file_path = Path(__file__).resolve()

# Get the parent directory of the current file
root_path = file_path.parent

# Add the root path to the sys.path list if it is not already there
if root_path not in sys.path:
    sys.path.append(str(root_path))

# Get the relative path of the root directory with respect to the current working directory
ROOT = root_path.relative_to(Path.cwd())

# Sources
IMAGE = 'Image'
VIDEO = 'Video'
WEBCAM = 'Webcam'
RTSP = 'RTSP'
YOUTUBE = 'YouTube'

SOURCES_LIST = [IMAGE, VIDEO, WEBCAM, RTSP, YOUTUBE]

# Images config
IMAGES_DIR = ROOT / 'images'
DEFAULT_IMAGE = IMAGES_DIR / 'ima1.jpg'
DEFAULT_DETECT_IMAGE = IMAGES_DIR / 'ppeimage.png'

# Videos config
VIDEO_DIR = ROOT / 'videos'
VIDEO_1_PATH = VIDEO_DIR / 'video1.mp4'
VIDEO_2_PATH = VIDEO_DIR / 'video2.mp4'
VIDEO_3_PATH = VIDEO_DIR / 'video3.mp4'
VIDEOS_DICT = {
    'video1': VIDEO_1_PATH,
    'video2': VIDEO_2_PATH,
    'video3': VIDEO_3_PATH,
}

# ML Model config
MODEL_DIR = ROOT / 'PPE-detection-system-for-workers-main'
DETECTION_MODEL = Path("C:/PPE-detection-system-for-workers-main/best.pt")

# In case of your custome model comment out the line above and
# Place your custom model pt file name at the line below 
# DETECTION_MODEL = MODEL_DIR / 'my_detection_model.pt'

SEGMENTATION_MODEL = MODEL_DIR / 'yolov8n-seg.pt'

# Webcam
WEBCAM_PATH = 0
