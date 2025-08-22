import cv2
import os

# --- IMPORTANT ---
# Disable the shared memory feature that can cause crashes in Docker
os.environ['OPENCV_X11_SHM_OUPUT'] = '0'

# --- Your RTSP Stream URL ---
rtsp_url = "rtsp://admin:qwer1234@192.168.10.30:554/Streaming/Channels/102"

print("Attempting to build GStreamer pipeline...")

# Define the GStreamer pipeline
pipeline = (
    f"rtspsrc location={rtsp_url} latency=0 ! "
    "rtph264depay ! h264parse ! nvv4l2decoder ! "
    "nvvidconv ! video/x-raw, format=BGRx ! "
    "videoconvert ! video/x-raw, format=BGR ! appsink max-buffers=1 drop=true"
)

print(f"Pipeline: {pipeline}")

print("\nAttempting to open video stream with cv2.VideoCapture...")
cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("\n--- !!! FAILED !!! ---")
    print("Error: cap.isOpened() returned False.")
    print("This means OpenCV could not open the GStreamer pipeline.")
    print("Possible reasons:")
    print("1. A GStreamer element is missing (e.g., rtph264depay).")
    print("2. Incorrect Docker permissions for NVIDIA runtime.")
else:
    print("\n--- SUCCESS ---")
    print("cap.isOpened() returned True. The pipeline was opened successfully.")
    print("Now, attempting to read the first frame...")

    ret, frame = cap.read()

    if not ret:
        print("\n--- !!! FAILED !!! ---")
        print("Error: cap.read() returned False.")
        print("The pipeline was opened, but no frame could be read.")
        print("Possible reasons:")
        print("1. Incorrect RTSP URL (wrong user/pass, IP, or path).")
        print("2. Network firewall is blocking the stream.")
        print("3. Video codec from camera is not H.264 or is otherwise incompatible.")
    else:
        print("\n--- !!! ULTIMATE SUCCESS !!! ---")
        print("A frame was successfully read from the stream!")
        print(f"Frame details: Shape={frame.shape}, Dtype={frame.dtype}")

print("\nReleasing capture object.")
cap.release()
print("Script finished.")
