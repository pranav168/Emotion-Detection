from numpy.testing._private.utils import suppress_warnings
import streamlit as st


from PIL import Image
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing import image as SImage
from keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import tempfile


# from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

DEMO_IMAGE = "Reports\demo.jpg"
DEMO_VIDEO = "Reports\demo-video.mp4"
classifier = load_model("Emotion-Detection\Models\my_model.h5")
face_classifier = cv2.CascadeClassifier(
    "C:\\Users\\pranav.singhal\\AppData\\Local\\Programs\\Python\\Python37\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml"
)
class_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]


@st.cache
def detect_emotion(image):
    # resize the frame to process it quickly
    frame = image

    labels = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y : y + h, x : x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            preds = classifier.predict(roi)[0]
            label = class_labels[preds.argmax()]

            label_position = (x, y)
            cv2.putText(
                frame,
                label,
                label_position,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (100, 255, 255),
                1,
            )
        else:
            cv2.putText(
                frame,
                "No Face Found",
                (10, 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                1,
            )

    return frame


st.title("Emotion Detection Application")

st.markdown(
    """
            * This model helps you detect emotions in an Image\n
            * So give an image with where the face is not blurry or disoriented 
            """
)

img_file_buffer = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if img_file_buffer is not None:
    image = np.array(Image.open(img_file_buffer))

else:
    image = np.array(Image.open(DEMO_IMAGE))


st.subheader("Original Image")

st.image(image, caption=f"Original Image", use_column_width=True, clamp=True)

emotion_analysis = detect_emotion(image)
st.markdown(
    """
            This model detects only the following emotions 
            **Angry,Happy,Neutral,Sad and Surprise**
            """
)


st.subheader("Emotion Analysis")

st.image(emotion_analysis, caption=f"detected Image", use_column_width=True)

st.subheader("Emotion Detection on Video")

video_file_buffer = st.file_uploader(
    "Upload a video", type=["mp4", "mov", "avi", "gif"]
)


tfflie = tempfile.NamedTemporaryFile(delete=False)


if not video_file_buffer:
    cap = cv2.VideoCapture(DEMO_VIDEO)

else:
    tfflie.write(video_file_buffer.read())

    cap = cv2.VideoCapture(tfflie.name)

stframe = st.empty()


while cap.isOpened():

    ret, frames = cap.read()

    labels = []

    gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frames, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y : y + h, x : x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            preds = classifier.predict(roi)[0]
            label = class_labels[preds.argmax()]

            label_position = (x, y)
            cv2.putText(
                frames,
                label,
                label_position,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
            )
        else:
            cv2.putText(
                frames,
                "No Face Found",
                (10, 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (255, 255, 0),
                2,
            )
    stframe.image(frames, channels="BGR", use_column_width=True)

