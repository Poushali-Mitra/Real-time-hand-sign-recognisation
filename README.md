# Real-time-Hand-Sign-Recognition
# Sign Language Recognition using TensorFlow, OpenCV, and MediaPipe

This project is a real-time **Sign Language Recognition** system that uses a Convolutional Neural Network (CNN) combined with **MediaPipe** for hand landmark detection. The system translates hand gestures into corresponding letters of the alphabet, forming words and sentences. Additionally, the system can speak the detected word or sentence using text-to-speech functionality.

## Features

- **Real-time sign language detection** using a webcam.
- **Prediction of letters (A-Z)** using a CNN-based model.
- **MediaPipe** for hand landmark detection and gesture recognition.
- **Word formation** by detecting consistent gestures over time.
- **Text-to-speech** using `pyttsx3` to read aloud the detected word.
- Ability to **add space** between words using the space bar.
- Ability to **remove the last character** using the backspace key.
- **Greeting message** when the application starts.hehe

## Installation

### Prerequisites

Make sure you have Python 3.10.0 installed on your system. To set up the project environment, follow these steps:

1. **Clone the repository**:

    ```bash
    git clone https://github.com/samratbose10/sign-langauge-to-text-and-text-to-speech.git
    ```

2. **Create a virtual environment**:

    If you're using Python:

    ```bash
    python -m venv myenv
    ```

3. **Activate the virtual environment**:

    - On Windows:

      ```bash
      myenv\Scripts\activate
      ```

    - On MacOS/Linux:

      ```bash
      source myenv/bin/activate
      ```

4. **Install dependencies**:

    Install all the necessary packages by running:

    ```bash
    pip install -r requirements.txt
    ```

    This will install:
    - `tensorflow`
    - `keras`
    - `mediapipe`
    - `opencv-python`
    - `pyttsx3`
    - `numpy`
    - `scikit-learn`
    - `pip install mediapipe --no-deps`
    - Other necessary libraries

## Usage

### Running the Application

1. **Train the Model**:
   
   Before running the real-time recognition system, you need to train the model using your dataset, and use webcam for the better results.

   - Prepare the dataset in the `dataset/` folder. The dataset should be structured with subdirectories for each letter (A-Z), each containing images of corresponding hand gestures.After running the imagecap.py (this is basically the capturing your gesture photo) you data will be saved in a folder called dataset and after that run python train_model.py . 

   - Run the following command to train the model:

     ```bash
     python train_model.py
     ```

     This will train the CNN model and save the best model as `hand_sign_landmark_model.h5`.

2. **Real-time Sign Language Recognition**:

   After the model is trained and saved as `hand_sign_landmark_model.h5`, you can run the real-time recognition system:

   ```bash
   python main.py
