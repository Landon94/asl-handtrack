import cv2
import time
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(1)

def main():

    hand = mp_hands.Hands()

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

                cv2.rectangle(rgb_frame,pt1=(x_min-30,y_min-30),pt2=(x_max+30,y_max+30),color=(0,255,0),thickness=2)

        final_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("image",cv2.flip(final_frame,1))

        if cv2.waitKey(1) & 0xFF == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()