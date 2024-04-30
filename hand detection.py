import cv2
import mediapipe as mp
import math
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

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

def store_features(features, label, database):
    database[label] = features

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
                finger_lengths = calculate_finger_lengths(hand_landmarks) ## location of fingers above
                for i, length in enumerate(finger_lengths):
                    cv2.putText(frame, f'Finger {i+1}: {length:.2f} pixels', (10, 30*(i+1)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Display the frame
        cv2.imshow('Hand Detection', frame)
        # print(finger_lengths)
        if cv2.waitKey(1) & 0xFF == ord('s'):
            cap.release()
            cv2.destroyAllWindows()
            return finger_lengths

def authenticate_user(features, database):
    if features is None:
        return None
    labels = list(database.keys())
    feature_list = np.array(list(database.values()))
    
    if len(feature_list) == 0:
        print("No users in database to match.")
        return None
    
    features_normalized = np.array(features).reshape(1, -1)
    
    print(features_normalized*10)
    feature_list_normalized = np.array(feature_list)
    print(feature_list_normalized*10)
    
    # Compute cosine similarity for all users in the database
    similarities = cosine_similarity(features_normalized*10, feature_list_normalized*10)
    print(similarities)
    
    # Find the index of the user with the highest similarity
    max_index = np.argmax(similarities)
    
    # Retrieve the label of the matching user
    matching_user_label = labels[max_index]
    
    # Check if the maximum similarity exceeds the threshold
    similarity_threshold = 0.9997
    if similarities[0, max_index] < similarity_threshold:
        return "No matching user found."
    
    return matching_user_label


database = {}

while True:
    action = input("Type 'register' to register, 'login' to authenticate, or 'exit' to quit: ").lower()
    if action == 'exit':
        break
    elif action == 'register':
        print("Registration:")
        user_name = input("Enter your name for registration: ")
        userhandimage = detect_hands()
        if userhandimage:
            store_features(userhandimage, user_name, database)
            print("success!")
        else:
            print("Failed to register. Please try again with a clear image of your hand.")
    elif action == 'login':
        print("Authentication: ")
        authen_hand_data = detect_hands()
        if authen_hand_data:
            authen_users = authenticate_user(authen_hand_data, database)
            if authen_users:
                print(f"Authenticated user ID: {authen_users}")
            else:
                print("Authentication Failed. Your hand aren't match.")
        else:
            print("Failed to authenticate. Please try again with a clear image of your hand.")
    else: 
        print("Invalid input. Please type 'register', 'login', or 'exit'.")
