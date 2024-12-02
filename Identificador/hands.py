# Import the necessary packages.
import cv2 as cv
import mediapipe as mp # To install this package: pip install mediapipe
import numpy as np

from keras.models import load_model

# Disable scientific notation for clarity.
np.set_printoptions(suppress=True)

# Load the model.
model = load_model("data/model/keras_model.h5", compile=False)
# Load the labels.
class_names = open("data/model/labels.txt", "r").readlines()

# Initialize the MediaPipe Hands model.
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Create a VideoCapture object.
video = cv.VideoCapture(0)

# Define the MediaPipe Hands parameters.
with mp_hands.Hands(
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    # Loop to read the video frames.
    while True:

        # Read the first frame.
        ret, frame = video.read()
        if not ret:
            break

        # Convert the frame to RGB.
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        # Process the frame.
        results = hands.process(frame_rgb)

        # Check if the hand was detected.
        if results.multi_hand_landmarks:

            # Loop to get the landmarks of each hand.
            for hand_landmarks in results.multi_hand_landmarks:

                # Draw the landmarks on the frame.
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

        # Resize the raw image into (224-height,224-width) pixels.
        resized = cv.resize(frame, (224, 224), interpolation=cv.INTER_AREA)
        resized = cv.cvtColor(resized, cv.COLOR_BGR2RGB)

        # Make the image a numpy array and reshape it to the models input shape.
        resized = np.asarray(resized, dtype=np.float32).reshape(1, 224, 224, 3)

        # Normalize the image array.
        resized = (resized / 127.5) - 1

        # Predicts the model.
        prediction = model.predict(resized)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]

        # Print prediction and confidence score.
        print("Class:", class_name[2:], end="")
        print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")
        cv.putText(frame, "Class: " + class_name[2:], (10, 50),
                    cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

        # Show the image in a window.
        cv.imshow("Frame", frame)
        if cv.waitKey(33) == ord("q"):
            break

video.release()
cv.destroyAllWindows()
