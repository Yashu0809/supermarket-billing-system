import cv2
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import numpy as np
import pandas as pd
import tensorflow as tf
import os
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Load the trained classification model and product metadata
MODEL_PATH = "model_trained/product_classifier.h5"
PRODUCTS_CSV = "sample products.csv"

model = tf.keras.models.load_model(MODEL_PATH)
product_data = pd.read_csv(PRODUCTS_CSV)

# Create the main GUI app
class ProductRecognizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Supermarket Billing System")
        self.root.geometry("900x600")

        self.cap = cv2.VideoCapture(0)

        self.video_label = tk.Label(self.root)
        self.video_label.pack()

        self.capture_button = tk.Button(self.root, text="Capture Product", command=self.capture_product)
        self.capture_button.pack(pady=10)

        self.product_info = tk.Label(self.root, text="", font=("Arial", 14))
        self.product_info.pack()

        self.bill = []
        self.total_amount = 0

        self.bill_listbox = tk.Listbox(self.root, height=10, width=60)
        self.bill_listbox.pack(pady=10)

        self.total_label = tk.Label(self.root, text="Total: ₹0", font=("Arial", 14, "bold"))
        self.total_label.pack()

        self.clear_button = tk.Button(self.root, text="Clear Bill", command=self.clear_bill)
        self.clear_button.pack(pady=5)

        self.update_video_stream()

    def update_video_stream(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
        self.root.after(10, self.update_video_stream)

    def preprocess_image(self, image):
        image = cv2.resize(image, (224, 224))
        image = preprocess_input(image)
        image = np.expand_dims(image, axis=0)
        return image

    def capture_product(self):
        ret, frame = self.cap.read()
        if not ret:
            messagebox.showerror("Error", "Failed to capture image")
            return

        processed_img = self.preprocess_image(frame)
        prediction = model.predict(processed_img)
        predicted_class = np.argmax(prediction)

        try:
            product_row = product_data.loc[predicted_class]
            name = product_row['Product']
            price = product_row['Price']

            self.bill.append((name, price))
            self.total_amount += price

            self.bill_listbox.insert(tk.END, f"{name} - ₹{price}")
            self.total_label.config(text=f"Total: ₹{self.total_amount}")
            self.product_info.config(text=f"Detected: {name} (₹{price})")

        except Exception as e:
            messagebox.showerror("Prediction Error", str(e))

    def clear_bill(self):
        self.bill = []
        self.total_amount = 0
        self.bill_listbox.delete(0, tk.END)
        self.total_label.config(text="Total: ₹0")
        self.product_info.config(text="")

def main():
    root = tk.Tk()
    app = ProductRecognizerApp(root)
    root.mainloop()

# ✅ Fixed line here
if __name__ == "__main__":
    main()
