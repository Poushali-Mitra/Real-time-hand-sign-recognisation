# import cv2
# import numpy as np
# from tkinter import Tk, Button, Label, Text, filedialog
# from PIL import Image, ImageTk
# import tensorflow as tf
# import mediapipe as mp
# import pyttsx3

# class SignLanguageApp:
#     def __init__(self, root):
#         self.root = root
#         self.root.title("Sign Language to Text")

#         self.sentence = ""
#         self.current_word = ""

#         self.engine = pyttsx3.init()

#         try:
#             self.model = tf.keras.models.load_model("hand_sign_landmark_model.h5")
#         except Exception as e:
#             print(f"Error loading model: {e}")
#             self.quit_app()

#         self.class_names = [
#             "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "Space"
#         ]

#         self.mp_hands = mp.solutions.hands
#         self.hands = self.mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

#         self.video_label = Label(self.root)
#         self.video_label.grid(row=0, column=0, columnspan=3)

#         self.text_box = Text(self.root, height=5, width=40)
#         self.text_box.grid(row=1, column=0, columnspan=3)

#         Button(self.root, text="Clear All", command=self.clear_text).grid(row=2, column=0)
#         Button(self.root, text="Save to a Text File", command=self.save_to_file).grid(row=2, column=1)
#         Button(self.root, text="Quit", command=self.quit_app).grid(row=2, column=2)

#         self.cap = cv2.VideoCapture(0)
#         if not self.cap.isOpened():
#             print("Error: Cannot access the webcam")
#             self.quit_app()
#             return

#         self.root.bind("<space>", self.add_space)
#         self.root.bind("s", self.speak_sentence)
#         self.root.bind("<BackSpace>", self.delete_last_character)

#         self.update_video_feed()

#     def clear_text(self):
#         self.text_box.delete(1.0, "end")
#         self.current_word = ""
#         self.sentence = ""

#     def save_to_file(self):
#         text = self.text_box.get(1.0, "end").strip()
#         if text:
#             file_path = filedialog.asksaveasfilename(defaultextension=".txt",
#                                                     filetypes=[("Text Files", "*.txt")])
#             if file_path:
#                 with open(file_path, "w") as f:
#                     f.write(text)

#     def quit_app(self):
#         if self.cap.isOpened():
#             self.cap.release()
#         self.root.destroy()

#     def extract_landmarks(self, frame):
#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         result = self.hands.process(rgb_frame)

#         if result.multi_hand_landmarks:
#             hand_landmarks = result.multi_hand_landmarks[0]
#             landmarks = []
#             for lm in hand_landmarks.landmark:
#                 landmarks.extend([lm.x, lm.y])
#             return np.array(landmarks).reshape(1, 21, 2)
#         return None

#     def predict_sign(self, frame):
#         landmarks = self.extract_landmarks(frame)
#         if landmarks is not None:
#             frame_resized = cv2.resize(frame, (64, 64))
#             frame_resized = frame_resized / 255.0
#             frame_resized = np.expand_dims(frame_resized, axis=0)

#             try:
#                 predictions = self.model.predict([frame_resized, landmarks])
#                 class_idx = np.argmax(predictions)
#                 return self.class_names[class_idx]
#             except Exception as e:
#                 print(f"Error during prediction: {e}")
#                 return "Error"
#         return "No hand detected"

#     def update_video_feed(self):
#         ret, frame = self.cap.read()
#         if not ret or frame is None:
#             print("Failed to grab frame")
#             self.quit_app()
#             return

#         frame = cv2.flip(frame, 1)

#         # Apply edge detection
#         gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         edges = cv2.Canny(gray_frame, 100, 200)

#         # Predict the sign
#         predicted_sign = self.predict_sign(frame)

#         if predicted_sign == "Space":
#             self.add_space(None)
#         elif predicted_sign not in ["No hand detected", "Error"]:
#             self.current_word = predicted_sign

#         self.text_box.delete(1.0, "end")
#         self.text_box.insert("end", self.sentence + self.current_word)

#         # Combine original frame and edges side by side for display
#         edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)  # Convert edges to RGB for consistency
#         combined_frame = np.hstack((frame, edges_colored))

#         frame_rgb = cv2.cvtColor(combined_frame, cv2.COLOR_BGR2RGB)
#         img = Image.fromarray(frame_rgb)
#         imgtk = ImageTk.PhotoImage(image=img)

#         self.video_label.imgtk = imgtk
#         self.video_label.configure(image=imgtk)

#         self.root.after(10, self.update_video_feed)

#     def add_space(self, event):
#         self.sentence += self.current_word + " "
#         self.current_word = ""

#     def delete_last_character(self, event):
#         if self.current_word:
#             self.current_word = self.current_word[:-1]
#         elif self.sentence:
#             self.sentence = self.sentence.rstrip()[:-1]

#     def speak_sentence(self, event):
#         self.engine.say(self.sentence.strip())
#         self.engine.runAndWait()

# if __name__ == "__main__":
#     root = Tk()
#     app = SignLanguageApp(root)
#     root.mainloop()
import cv2
import numpy as np
from tkinter import Tk, Button, Label, Text, filedialog
from PIL import Image, ImageTk
import tensorflow as tf
import mediapipe as mp
import pyttsx3

class SignLanguageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sign Language to Text")

        self.sentence = ""
        self.current_word = ""

        self.engine = pyttsx3.init()

        try:
            self.model = tf.keras.models.load_model("hand_sign_landmark_model.h5")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.quit_app()

        self.class_names = [
            "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "Space"
        ]

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

        self.video_label = Label(self.root)
        self.video_label.grid(row=0, column=0, columnspan=3)

        self.threshold_label = Label(self.root)
        self.threshold_label.grid(row=0, column=3, columnspan=3)

        self.text_box = Text(self.root, height=5, width=40)
        self.text_box.grid(row=1, column=0, columnspan=3)

        Button(self.root, text="Clear All", command=self.clear_text).grid(row=2, column=0)
        Button(self.root, text="Save to a Text File", command=self.save_to_file).grid(row=2, column=1)
        Button(self.root, text="Quit", command=self.quit_app).grid(row=2, column=2)

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Cannot access the webcam")
            self.quit_app()
            return

        self.root.bind("<space>", self.add_space)
        self.root.bind("s", self.speak_sentence)
        self.root.bind("<BackSpace>", self.delete_last_character)

        self.update_video_feed()

    def clear_text(self):
        self.text_box.delete(1.0, "end")
        self.current_word = ""
        self.sentence = ""

    def save_to_file(self):
        text = self.text_box.get(1.0, "end").strip()
        if text:
            file_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text Files", "*.txt")])
            if file_path:
                with open(file_path, "w") as f:
                    f.write(text)

    def quit_app(self):
        if self.cap.isOpened():
            self.cap.release()
        self.root.destroy()

    def extract_landmarks(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb_frame)

        if result.multi_hand_landmarks:
            hand_landmarks = result.multi_hand_landmarks[0]
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y])
            return np.array(landmarks).reshape(1, 21, 2)
        return None

    def predict_sign(self, frame):
        landmarks = self.extract_landmarks(frame)
        if landmarks is None:
            return "No hands detected"

        frame_resized = cv2.resize(frame, (64, 64))
        frame_resized = frame_resized / 255.0
        frame_resized = np.expand_dims(frame_resized, axis=0)

        try:
            predictions = self.model.predict([frame_resized, landmarks])
            class_idx = np.argmax(predictions)
            return self.class_names[class_idx]
        except Exception as e:
            print(f"Error during prediction: {e}")
            return "Error"

    def update_video_feed(self):
        ret, frame = self.cap.read()
        if not ret or frame is None:
            print("Failed to grab frame")
            self.quit_app()
            return

        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape
        roi_x1, roi_y1, roi_x2, roi_y2 = width // 4, height // 4, 3 * width // 4, 3 * height // 4
        roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]

        gray_frame = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        adaptive_thresh = cv2.adaptiveThreshold(gray_frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                cv2.THRESH_BINARY, 11, 2)
        edges = cv2.Canny(adaptive_thresh, 50, 150)

        predicted_sign = self.predict_sign(roi)

        if predicted_sign == "No hands detected":
            cv2.putText(frame, "No hands detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            self.current_word = ""
        elif predicted_sign == "Space":
            self.add_space(None)
        elif predicted_sign != "Error":
            self.current_word = predicted_sign

        self.text_box.delete(1.0, "end")
        self.text_box.insert("end", self.sentence + self.current_word)

        cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 0), 2)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)

        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        threshold_rgb = cv2.cvtColor(adaptive_thresh, cv2.COLOR_GRAY2RGB)
        threshold_img = Image.fromarray(threshold_rgb)
        threshold_imgtk = ImageTk.PhotoImage(image=threshold_img)
        self.threshold_label.imgtk = threshold_imgtk
        self.threshold_label.configure(image=threshold_imgtk)

        self.root.after(10, self.update_video_feed)

    def add_space(self, event):
        self.sentence += self.current_word + " "
        self.current_word = ""

    def speak_sentence(self, event):
        self.engine.say(self.sentence.strip())
        self.engine.runAndWait()

if __name__ == "__main__":
    root = Tk()
    app = SignLanguageApp(root)
    root.mainloop()



