import os
import numpy as np
import cv2
import red_neuronal as rn

import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

fashion_mnist_labels = {
    0: 'T-shirt/top',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot'
}


def build_model_from_params(params_path, input_size=784):
    model = rn.Model()
    model.add(rn.Layer_Dense(input_size, 128))
    model.add(rn.Activation_ReLU())
    model.add(rn.Layer_Dense(128, 128))
    model.add(rn.Activation_ReLU())
    model.add(rn.Layer_Dense(128, 10))
    model.add(rn.Activation_Softmax())
    model.finalize()
    model.load_parameters(params_path)
    return model


def preprocess_image_array(img):
    # img: grayscale numpy array
    img = cv2.resize(img, (28, 28))
    img = 255 - img
    arr = (img.astype(np.float32).reshape(1, -1) - 127.5) / 127.5
    return arr


def preprocess_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return preprocess_image_array(img)


class InferenceApp:
    def __init__(self, root):
        self.root = root
        root.title('Fashion-MNIST Inference')

        self.model = None
        self.model_path = None
        self.params_path = None
        self.image_path = None
        self.preview_imgtk = None

        btn_frame = tk.Frame(root)
        btn_frame.pack(padx=8, pady=8, fill='x')

        tk.Button(btn_frame, text='Load Model (.model)', command=self.load_model).pack(side='left')
        tk.Button(btn_frame, text='Load Params (.parms)', command=self.load_params).pack(side='left')
        tk.Button(btn_frame, text='Load Image', command=self.load_image).pack(side='left')
        tk.Button(btn_frame, text='Predict', command=self.predict).pack(side='left')

        info_frame = tk.Frame(root)
        info_frame.pack(padx=8, pady=4, fill='x')

        self.model_label = tk.Label(info_frame, text='Model: None')
        self.model_label.pack(anchor='w')
        self.image_label = tk.Label(info_frame, text='Image: None')
        self.image_label.pack(anchor='w')

        self.canvas = tk.Canvas(root, width=200, height=200, bg='white')
        self.canvas.pack(padx=8, pady=8)

        self.result_label = tk.Label(root, text='Prediction: -', font=('Arial', 14))
        self.result_label.pack(padx=8, pady=4)

        self.conf_text = tk.Text(root, height=6, width=40)
        self.conf_text.pack(padx=8, pady=4)

    def load_model(self):
        path = filedialog.askopenfilename(title='Select model file', filetypes=[('Model files', '*.model'), ('All files', '*.*')])
        if not path:
            return
        try:
            self.model = rn.Model.load(path)
            self.model_path = path
            self.model_label.config(text=f'Model: {os.path.basename(path)}')
            messagebox.showinfo('Model loaded', f'Loaded model from {path}')
        except Exception as e:
            messagebox.showerror('Error', f'Failed to load model: {e}')

    def load_params(self):
        path = filedialog.askopenfilename(title='Select parameters file', filetypes=[('Params', '*.parms'), ('All files', '*.*')])
        if not path:
            return
        try:
            self.model = build_model_from_params(path)
            self.params_path = path
            self.model_label.config(text=f'Params: {os.path.basename(path)}')
            messagebox.showinfo('Params loaded', f'Loaded parameters from {path}')
        except Exception as e:
            messagebox.showerror('Error', f'Failed to load parameters: {e}')

    def load_image(self):
        path = filedialog.askopenfilename(title='Select image', filetypes=[('Images', '*.png;*.jpg;*.jpeg;*.bmp'), ('All files', '*.*')])
        if not path:
            return
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            messagebox.showerror('Error', 'Failed to read image')
            return
        self.image_path = path
        self.image_label.config(text=f'Image: {os.path.basename(path)}')

        # Prepare preview: invert and resize for display
        preview = cv2.resize(255 - img, (200, 200))
        preview_pil = Image.fromarray(preview).convert('L')
        self.preview_imgtk = ImageTk.PhotoImage(preview_pil)
        self.canvas.create_image(0, 0, anchor='nw', image=self.preview_imgtk)

    def predict(self):
        if self.model is None:
            messagebox.showwarning('No model', 'Load a model (.model) or parameters (.parms) first')
            return
        if self.image_path is None:
            messagebox.showwarning('No image', 'Load an image first')
            return
        try:
            x = preprocess_image(self.image_path)
            confidences = self.model.predict(x)[0]
            preds = self.model.output_layer_activation.predictions(confidences.reshape(1, -1))
            label = fashion_mnist_labels.get(int(preds[0]), str(int(preds[0])))
            self.result_label.config(text=f'Prediction: {label}')
            # Show top 5 confidences
            top_idx = np.argsort(confidences)[::-1][:5]
            self.conf_text.delete('1.0', tk.END)
            for i in top_idx:
                self.conf_text.insert(tk.END, f'{i}: {fashion_mnist_labels.get(int(i), str(i))} -> {confidences[i]:.4f}\n')
        except Exception as e:
            messagebox.showerror('Error', f'Inference failed: {e}')


def main():
    root = tk.Tk()
    app = InferenceApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()
