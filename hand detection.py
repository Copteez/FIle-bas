import cv2
import mediapipe as mp
import math

def calculate_finger_lengths(hand_landmarks):
    # Finger landmarks indices
    finger_indices = [[4, 3, 2, 1],  # Thumb (indices of landmarks from the wrist to fingertip)
                      [8, 7, 6, 5],  # Index finger
                      [12, 11, 10, 9],  # Middle finger
                      [16, 15, 14, 13],  # Ring finger
                      [20, 19, 18, 17]]  # Little finger

    finger_lengths = []

    for finger_index in finger_indices:
        # Calculate the length of the finger segments
        length = 0
        for i in range(len(finger_index) - 1):
            x1, y1 = hand_landmarks.landmark[finger_index[i]].x, hand_landmarks.landmark[finger_index[i]].y
            x2, y2 = hand_landmarks.landmark[finger_index[i+1]].x, hand_landmarks.landmark[finger_index[i+1]].y
            length += math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        finger_lengths.append(length)
    
    return finger_lengths

def detect_hands():
    cap = cv2.VideoCapture(0)  # Open the default camera
    hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Get frame dimensions
        height, width, _ = frame.shape
        # Calculate center coordinates
        center_x, center_y = int(width / 2), int(height / 2)
        # Define box dimensions
        box_width, box_height = 400, 400
        # Calculate box coordinates
        x1, y1 = center_x - int(box_width / 2), center_y - int(box_height / 2)
        x2, y2 = center_x + int(box_width / 2), center_y + int(box_height / 2)
        # Draw box on frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Convert the image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame
        results = hands.process(rgb_frame)

        # Draw hand landmarks if hands are detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks on the frame
                for landmark in hand_landmarks.landmark:
                    x = int(landmark.x * width)
                    y = int(landmark.y * height)
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

                # Calculate finger lengths
                finger_lengths = calculate_finger_lengths(hand_landmarks)
                for i, length in enumerate(finger_lengths):
                    cv2.putText(frame, f'Finger {i+1}: {length:.2f} pixels', (10, 30*(i+1)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Display the frame
        cv2.imshow('Hand Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_hands()
