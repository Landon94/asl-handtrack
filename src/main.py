import cv2
import time
import mediapipe as mp
import numpy as np
import tensorflow as tf
from model import create_model

CLASS_NAMES = [
    "A","B","Blank","C","D","E","F","G","H","I","J",
    "K","L","M","N","O","P","Q","R","S","T",
    "U","V","W","X","Y","Z",
]


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(1)

def main():

    hand = mp_hands.Hands()

    model = create_model((180,180,3))
    model.load_weights("asl_cnn.h5")

    while True:

        ret,frame = cap.read()

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        result = hand.process(rgb_frame)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(rgb_frame,hand_landmarks,
                                        mp_hands.HAND_CONNECTIONS,
                                        mp_drawing_styles.get_default_hand_landmarks_style(),
                                        mp_drawing_styles.get_default_hand_connections_style()
                                        )
                
                height, width, _ = frame.shape
                xs = [landmark.x * width for landmark in hand_landmarks.landmark]
                ys = [landmark.y * height for landmark in hand_landmarks.landmark]
                
                x_min, x_max = int(min(xs)), int(max(xs))
                y_min, y_max = int(min(ys)), int(max(ys))

                cropped_roi = frame[y_min-100:y_max+100, x_min-100:x_max+100]

                roi_rgb = cv2.cvtColor(cropped_roi, cv2.COLOR_BGR2RGB)

                resized_frame = cv2.resize(roi_rgb,(180,180),interpolation=cv2.INTER_AREA)
                resized_frame = np.expand_dims(resized_frame, axis=0)

                cv2.imshow("ROI", cropped_roi)
    
                preds = model.predict(resized_frame)
                cls_idx = int(np.argmax(preds[0]))
                conf = float(np.max(preds[0]))
                # print(predictions)

                label = CLASS_NAMES[cls_idx]
                # print(label)
                # print(f"{cls_idx=} {conf=:.3f}")

                cv2.rectangle(rgb_frame,pt1=(x_min-30,y_min-30),pt2=(x_max+30,y_max+30),color=(0,255,0),thickness=2)
                cv2.putText(rgb_frame,label,(x_min-50,y_min-50),cv2.FONT_HERSHEY_SIMPLEX,2.0,(255,0,0),3,cv2.LINE_AA)

        final_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("image",cv2.flip(final_frame,1))

        if cv2.waitKey(1) & 0xFF == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()