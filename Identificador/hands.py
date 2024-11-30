# Import the necessary packages.
import cv2 as cv
import mediapipe as mp # To install this package: pip install mediapipe


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

        # Show the first frame.
        cv.imshow("Frame", frame)
        if cv.waitKey(33) == ord("q"):
            break

video.release()
cv.destroyAllWindows()
