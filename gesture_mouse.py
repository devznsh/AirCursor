import cv2
import mediapipe as mp
import pyautogui
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
    model_complexity=0  # Faster processing
)

# Screen and mouse settings
screen_w, screen_h = pyautogui.size()
smoothening = 3
prev_x, prev_y = 0, 0

cap = cv2.VideoCapture(0)
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue
    
    # Flip and convert to RGB
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process with MediaPipe
    results = hands.process(frame_rgb)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get index finger tip (landmark 8)
            lm = hand_landmarks.landmark[8]
            x = np.interp(lm.x, [0.1, 0.9], [0, screen_w])
            y = np.interp(lm.y, [0.1, 0.9], [0, screen_h])
            
            # Smooth cursor movement
            curr_x = prev_x + (x - prev_x) / smoothening
            curr_y = prev_y + (y - prev_y) / smoothening
            pyautogui.moveTo(curr_x, curr_y)
            prev_x, prev_y = curr_x, curr_y
            
            # Click detection (thumb-index distance)
            thumb_tip = hand_landmarks.landmark[4]
            distance = ((lm.x - thumb_tip.x)**2 + (lm.y - thumb_tip.y)**2)**0.5
            if distance < 0.03:
                pyautogui.click()
                cv2.circle(frame, (int(lm.x*frame.shape[1]), int(lm.y*frame.shape[0])), 
                          10, (0,255,0), -1)

    # Display FPS
    fps = cap.get(cv2.CAP_PROP_FPS)
    cv2.putText(frame, f"FPS: {int(fps)}", (10,30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    
    cv2.imshow('Gesture Mouse (MediaPipe)', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()