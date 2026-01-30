import numpy as np # help in converting image in binary format(0,1)
import cv2 # open cv library help in image capture processing etc
import os
import mediapipe as mp # help in landmark detection 
from sklearn.preprocessing import MinMaxScaler # used in machine learning task

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) # one hand at a time , wait for some time , take static image 

DATASET_DIR = 'dataset_3'
letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


for letter in letters:
    os.makedirs(os.path.join(DATASET_DIR, letter), exist_ok=True) # create a directiory of list

cap = cv2.VideoCapture(1) # video capture 

if not cap.isOpened():
    print("Error: Could not open video capture.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280) 
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  

print("Press the key corresponding to the letter you want to capture images for (A-Z).")
print("Press esc to quit.")

def normalize_image(image):# diffrent pixels and value range to equal 
    scaler = MinMaxScaler()
    normalized_image = scaler.fit_transform(image.reshape(-1, image.shape[-1])).reshape(image.shape)
    return (normalized_image * 255).astype(np.uint8)

def resize_image(image):
    resized_image = cv2.resize(image, (224, 224))
    return resized_image

def apply_data_augmentation(image): # scaled image zoom in and translated shift to a side 
    
    scaled_image = cv2.resize(image, (int(ROI_WIDTH * 1.2), int(ROI_HEIGHT * 1.2)))
    
    translated_image = cv2.warpAffine(image, np.float32([[1, 0, 50], [0, 1, 50]]), (ROI_WIDTH, ROI_HEIGHT))
    return scaled_image, translated_image


def advanced_segmentation(image): # help in detecting contours in bitwise and applying mask o original image segmentation
    
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    mask = np.zeros(image.shape, dtype=np.uint8)
    for contour in contours:
        cv2.drawContours(mask, [contour], -1, 255, -1)
    
    segmented_image = cv2.bitwise_and(image, mask)
    return segmented_image


x_start, y_start, ROI_WIDTH, ROI_HEIGHT = 50, 80, 500, 500 

while True:
    ret, frame = cap.read() 
    if not ret:
        print("Failed to capture image. Exiting...")
        break

    frame = cv2.flip(frame, 1) 
    h, w, _ = frame.shape

   
    ROI = frame[y_start:y_start + ROI_HEIGHT, x_start:x_start + ROI_WIDTH]

   
    cv2.rectangle(frame, (x_start, y_start), (x_start + ROI_WIDTH, y_start + ROI_HEIGHT), (255, 255, 255), 2)

    # roi region of frame x coordinates and y coordinates 
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if ROI is not None:
        
        gray_frame = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)
        blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 2)
        adaptive_threshold = cv2.adaptiveThreshold(blurred_frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        ret, thresh = cv2.threshold(adaptive_threshold, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        segmented_image = advanced_segmentation(thresh)# darkens the edges for prominent hand detail capture

       
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for landmark in hand_landmarks.landmark:
                    
                    cx = int((landmark.x * w) - x_start)# convert x coordinate to actual pixel value of frame width
                    cy = int((landmark.y * h) - y_start)# convert the y coordinate to height
                    if 0 <= cx < ROI_WIDTH and 0 <= cy < ROI_HEIGHT:
                        cv2.circle(segmented_image, (cx, cy), 5, (0, 255, 0), -1)

        cv2.imshow('Hand Gesture Capture', segmented_image)  

    cv2.imshow('Frame', frame) 

    key = cv2.waitKey(1) & 0xFF # wait for 1 millisec

    if key == 27: 
        break

    if chr(key).upper() in letters: # convert character to assic code and to upper case
        letter = chr(key).upper()  
        letter_dir = os.path.join(DATASET_DIR, letter)

        os.makedirs(letter_dir, exist_ok=True)# check the presence of directory and return true

        
        try:# block in pyton helps in exception handling
            
            normalized_image = normalize_image(segmented_image)
            resized_image = resize_image(normalized_image)

           
            image_save_path = os.path.join(letter_dir, f'{letter}_{len(os.listdir(letter_dir))}.png')
            cv2.imwrite(image_save_path, resized_image)
            print(f"Saved image for letter '{letter}' at {image_save_path}")

            
            npy_save_path = os.path.join(letter_dir, f'{letter}_{len(os.listdir(letter_dir))}.npy')
            np.save(npy_save_path, resized_image)
            print(f"Saved NumPy file for letter '{letter}' at {npy_save_path}")

           
            if results.multi_hand_landmarks:
                landmarks = []
                for hand_landmarks in results.multi_hand_landmarks:
                    landmarks.append([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
                landmarks_save_path = os.path.join(letter_dir, f'{letter}_{len(os.listdir(letter_dir))}_landmarks.npy')
                np.save(landmarks_save_path, landmarks)
                print(f"Saved landmarks for letter '{letter}' at {landmarks_save_path}")

            
            augmented_images = apply_data_augmentation(segmented_image)
            for idx, aug_image in enumerate(augmented_images):
                aug_image_save_path = os.path.join(letter_dir, f'{letter}_aug_{idx}.png')
                cv2.imwrite(aug_image_save_path, aug_image)
                print(f"Saved augmented image for letter '{letter}' at {aug_image_save_path}")

        except Exception as e:
            print(f"Error saving data for letter '{letter}': {e}")
          
          
cap.release() # relese web cam
cv2.destroyAllWindows()# close open cv