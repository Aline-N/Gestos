import cv2 as cv
import mediapipe as mp 
import numpy as np

from keras.models import load_model


np.set_printoptions(suppress=True)


model = load_model("data/model/keras_model.h5", compile=False)

class_names = open("data/model/labels.txt", "r").readlines()


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


video = cv.VideoCapture(0)


with mp_hands.Hands(
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    
    while True:

        
        ret, frame = video.read()
        if not ret:
            break

        
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        
        results = hands.process(frame_rgb)

        
        if results.multi_hand_landmarks:

            
            for hand_landmarks in results.multi_hand_landmarks:

                
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )


        resized = cv.resize(frame, (224, 224), interpolation=cv.INTER_AREA)
        resized = cv.cvtColor(resized, cv.COLOR_BGR2RGB)

        
        resized = np.asarray(resized, dtype=np.float32).reshape(1, 224, 224, 3)

        
        resized = (resized / 127.5) - 1


        prediction = model.predict(resized)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]

        
        print("Class:", class_name[2:], end="")
        print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")
        cv.putText(frame, "Class: " + class_name[2:], (10, 50),
                    cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

        
        cv.imshow("Frame", frame)
        if cv.waitKey(33) == ord("q"):
            break

video.release()
cv.destroyAllWindows()
