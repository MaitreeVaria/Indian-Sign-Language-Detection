import cv2
import mediapipe as mp
import copy
import itertools
from tensorflow import keras
import numpy as np
import pandas as pd
import string
# load the saved model from file
model = keras.models.load_model("model_alldatatest1.h5")
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

alphabet =  ['1','2','3','4','5','6','7','8','9']
alphabet += list(string.ascii_uppercase)
# functions
def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)


        landmark_point.append([landmark_x, landmark_y])

    return landmark_point

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    model_complexity=0,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    image = cv2.flip(image, 1)
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    debug_image = copy.deepcopy(image)
    xmin, ymin, xmax, ymax = image.shape[1], image.shape[0], 0, 0
    if results.multi_hand_landmarks:
      for hand_landmarks, handedness in zip(results.multi_hand_landmarks,results.multi_handedness):
        landmark_list = calc_landmark_list(debug_image, hand_landmarks)
        # Conversion to relative coordinates / normalized coordinates
        pre_processed_landmark_list = pre_process_landmark(landmark_list)

 
        # Get the coordinates of the hand landmarks
        landmark_x = [landmark.x for landmark in hand_landmarks.landmark]
        landmark_y = [landmark.y for landmark in hand_landmarks.landmark]
        x, y, w, h = int(min(landmark_x) * image.shape[1]), int(min(landmark_y) * image.shape[0]), \
                     int((max(landmark_x) - min(landmark_x)) * image.shape[1]), \
                     int((max(landmark_y) - min(landmark_y)) * image.shape[0])
        # Update the bounding box coordinates to include both hands
        xmin, ymin = min(xmin, x), min(ymin, y)
        xmax, ymax = max(xmax, x + w), max(ymax, y + h)

        # Draw a bounding box around both hands
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

    
    
        df = pd.DataFrame(pre_processed_landmark_list).transpose()

        predictions = model.predict(df, verbose = 0)
        predicted_classes = np.argmax(predictions, axis=1)
        label = alphabet[predicted_classes[0]]
        global frame 
        frame = cv2.flip(image, 1)
        # Write the hand prediction label on top of the bounding box
        cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        print(alphabet[predicted_classes[0]])
        print("------------------------")
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
