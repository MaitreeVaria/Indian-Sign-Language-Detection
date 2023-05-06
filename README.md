

## Indian Sign Language Detection using Mediapipe

This project is aimed at detecting and recognizing Indian Sign Language (ISL) gestures using the Mediapipe library. The project is implemented in Python.

### Requirements

To run this project, you will need the following dependencies:

- Python 3.6 or higher
- Mediapipe library
- OpenCV library
- Numpy library

### Installation

1. Install Python 3.6 or higher on your system.
2. Install the Mediapipe library using the following command:

   ```
   pip install mediapipe
   ```

3. Install the OpenCV library using the following command:

   ```
   pip install opencv-python
   ```

4. Install the Numpy library using the following command:

   ```
   pip install numpy
   ```

### Usage

1. Clone the repository to your local machine.

2. Open the command prompt and navigate to the cloned directory.

3. Run the following command to start the program:

   ```
   python isl_mediapipe.py
   ```

4. The program will start and display the video stream from the webcam.

5. To exit the program, press the 'q' key.

### How it works

The program uses the Mediapipe library to detect landmarks on the hand and fingers of the user. These landmarks are used to determine the position and movement of the hand, which is then used to recognize various ISL gestures.

The program uses a pre-trained model to recognize the ISL gestures. The model is trained on a dataset of hand gestures captured using the Mediapipe library.

### Future Improvements

The following improvements can be made to the project:

- Improve the accuracy of gesture recognition by training the model on a larger dataset.
- Add support for more ISL gestures.
- Implement a feature to convert the recognized gestures into text or speech.
- Make the program more user-friendly by adding a GUI.


