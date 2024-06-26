import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.mixture import GaussianMixture
import copy

def get_image_input():
    choice = input("Choose input method - Type 'upload' to upload an image or 'camera' to use the camera: ")
    if choice.lower() == 'upload':
        file_path = input("Enter the path to your image file: ").strip().strip('"')
        image = cv2.imread(file_path)
        if image is None:
            print("Failed to load image. Please check the file path and ensure it is correct.")
            return get_image_input()
        return image
    elif choice.lower() == 'camera':
        return capture_hand_image()
    else:
        print("Invalid input. Please choose 'upload' or 'camera'.")
        return get_image_input()

def capture_hand_image():
    cap = cv2.VideoCapture(0) 
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    print("Please place your hand in the camera frame and press 's' to save the image.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        cv2.imshow('Hand Capture', frame)
        if cv2.waitKey(1) == ord('s'):
            print("Image captured")
            cv2.imwrite('test_image.jpg', frame)
            break
    cap.release()
    cv2.destroyAllWindows()
    return frame


def preprocess_image(image):
    img = cv2.resize(image, None, fx=0.5, fy=0.5)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    cv2.imwrite('hsv_image.jpg', hsv)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=3)
    mask = cv2.GaussianBlur(mask, (5,5), 100)
    cv2.imwrite('mask_image.jpg', mask)
    return mask


def extract_features(image):
    processed_image = preprocess_image(image)
    contours, _ = cv2.findContours(processed_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours and len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) < 1000:
            print("No valid hand detected in the image.")
            return None
        moments = cv2.moments(largest_contour)
        huMoments = cv2.HuMoments(moments)
        for i in range(0, 7):
            huMoments[i] = -1 * np.sign(huMoments[i]) * np.log10(np.abs(huMoments[i]))
        return huMoments.flatten().tolist()
    else:
        print("No hand found in the image.")
        return None

def store_features(features, label, database):
    database[label] = features

def calculate_angle(far, start, end):
    a = np.linalg.norm(far - start)
    b = np.linalg.norm(far - end)
    c = np.linalg.norm(start - end)
    angle = np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))
    angle = np.degrees(angle)
    return angle

def authenticate_user(features, database):
    if features is None:
        return None
    labels = list(database.keys())
    feature_list = np.array(list(database.values()))
    if len(feature_list) == 0:
        print("No users in database to match.")
        return None
    
    features_normalized = np.array(features).reshape(1, -1)
    feature_list_normalized = np.array(feature_list)
    
    similarities = cosine_similarity(features_normalized, feature_list_normalized)
    max_similarity = np.max(similarities)
    
    similarity_threshold = 0.5  
    
    if max_similarity < similarity_threshold:
        return "No matching user found."
    
    max_index = np.argmax(similarities)
    return labels[max_index]


database = {}

while True:
    action = input("Type 'register' to register, 'login' to authenticate, or 'quit' to quit: ").lower()
    if action == 'exit':
        break
    elif action == 'register':
        print("Registration:")
        user_name = input("Enter your name for registration: ")
        user_hand_image = get_image_input()
        hand_features = extract_features(user_hand_image)
        if hand_features:
            store_features(hand_features, user_name, database)
            print("Registration successful.")
        else:
            print("Failed to register. Please try again with a clear image of your hand.")
    elif action == 'login':
        print("Authentication:")
        auth_hand_image = get_image_input()
        auth_features = extract_features(auth_hand_image)
        if auth_features:
            authenticated_user = authenticate_user(auth_features, database)
            if authenticated_user:
                print(f"Authenticated user ID: {authenticated_user}")
            else:
                print("Authentication failed. No matching user found.")
        else:
            print("Failed to authenticate. Please try again with a clear image of your hand.")
    else:
        print("Invalid input. Please type 'register', 'login', or 'exit'.")
