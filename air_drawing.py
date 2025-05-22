import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7
)

# Drawing settings
canvas = None
prev_point = None
colors = [(0,255,0), (255,0,0), (0,0,255)]  # Green, Red, Blue
current_color = 0
brush_size = 8

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue
    
    # Initialize canvas
    if canvas is None:
        canvas = np.zeros_like(frame)
    
    # Flip and process
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get index finger tip
            lm = hand_landmarks.landmark[8]
            x, y = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
            
            # Draw line
            if prev_point:
                cv2.line(canvas, prev_point, (x,y), colors[current_color], brush_size)
            prev_point = (x, y)
    
    # Combine frame and canvas
    frame = cv2.addWeighted(frame, 0.7, canvas, 0.3, 0)
    
    # UI
    cv2.putText(frame, "Air Drawing Mode", (10,30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
    cv2.putText(frame, f"Color: {['Green','Red','Blue'][current_color]}", (10,60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    
    # Controls
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('c'):
        canvas.fill(0)  # Clear
    elif key == ord('b'):
        current_color = (current_color + 1) % 3  # Cycle colors
    
    cv2.imshow('Air Drawing', frame)

cap.release()
cv2.destroyAllWindows()