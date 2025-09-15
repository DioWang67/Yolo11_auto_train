import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from tkinter import colorchooser
from PIL import Image, ImageTk
import os

class MaskEditor:
    def __init__(self, normal_img_path, defect_img_path):
        # 讀取圖片
        self.normal_img = cv2.imread(normal_img_path, cv2.IMREAD_GRAYSCALE)
        self.defect_img = cv2.imread(defect_img_path, cv2.IMREAD_GRAYSCALE)

        if self.normal_img.shape != self.defect_img.shape:
            self.defect_img = cv2.resize(self.defect_img, (self.normal_img.shape[1], self.normal_img.shape[0]))

        self.normal_img = cv2.equalizeHist(self.normal_img)
        self.defect_img = cv2.equalizeHist(self.defect_img)

        self.diff = cv2.absdiff(self.normal_img, self.defect_img)

        self.threshold = 30
        self.generate_mask()

        # 編輯模式參數
        self.erasing = False
        self.drawing = False
        self.brush_size = 20
        self.mask_color = (0, 0, 255)  # 初始為紅色遮罩

        self.root = tk.Tk()
        self.root.title("Mask Editor")

        self.display_img = cv2.cvtColor(self.defect_img, cv2.COLOR_GRAY2BGR)
        self.photo = ImageTk.PhotoImage(image=Image.fromarray(self.display_img))
        self.canvas = tk.Canvas(self.root, width=self.display_img.shape[1], height=self.display_img.shape[0])
        self.canvas_img_id = self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        self.canvas.pack()
        self.update_overlay()

        # 閾值滑動條
        self.threshold_label = tk.Label(self.root, text=f"Threshold: {self.threshold}")
        self.threshold_label.pack()
        self.threshold_slider = ttk.Scale(self.root, from_=0, to_=100, orient=tk.HORIZONTAL, command=self.update_threshold)
        self.threshold_slider.set(self.threshold)
        self.threshold_slider.pack()

        # 功能按鈕
        self.erase_button = tk.Button(self.root, text="Toggle Eraser", command=self.toggle_eraser)
        self.erase_button.pack()

        self.draw_button = tk.Button(self.root, text="Toggle Pen", command=self.toggle_draw)
        self.draw_button.pack()

        self.color_button = tk.Button(self.root, text="Choose Mask Color", command=self.choose_color)
        self.color_button.pack()

        self.save_button = tk.Button(self.root, text="Save Mask", command=self.save_mask)
        self.save_button.pack()

        # 滑鼠事件綁定
        self.canvas.bind("<B1-Motion>", self.handle_mouse)
        self.canvas.bind("<Button-1>", self.handle_mouse)

        self.root.mainloop()

    def generate_mask(self):
        _, self.binary_mask = cv2.threshold(self.diff, self.threshold, 255, cv2.THRESH_BINARY)
        kernel = np.ones((5, 5), np.uint8)
        self.binary_mask = cv2.dilate(self.binary_mask, kernel, iterations=1)
        self.binary_mask = cv2.erode(self.binary_mask, kernel, iterations=1)

    def update_threshold(self, val):
        self.threshold = int(float(val))
        self.threshold_label.config(text=f"Threshold: {self.threshold}")
        self.generate_mask()
        self.update_overlay()

    def update_overlay(self):
        self.display_img = cv2.cvtColor(self.defect_img, cv2.COLOR_GRAY2BGR)
        mask_colored = np.zeros_like(self.display_img)
        b, g, r = self.mask_color
        mask_colored[:, :, 0] = (self.binary_mask // 255) * b
        mask_colored[:, :, 1] = (self.binary_mask // 255) * g
        mask_colored[:, :, 2] = (self.binary_mask // 255) * r

        self.display_img = cv2.addWeighted(self.display_img, 0.7, mask_colored, 0.3, 0)
        self.photo = ImageTk.PhotoImage(image=Image.fromarray(self.display_img))
        self.canvas.itemconfig(self.canvas_img_id, image=self.photo)

    def toggle_eraser(self):
        self.erasing = not self.erasing
        if self.erasing:
            self.drawing = False
        self.erase_button.config(
            text="Disable Eraser" if self.erasing else "Enable Eraser"
        )

    def toggle_draw(self):
        self.drawing = not self.drawing
        if self.drawing:
            self.erasing = False
        self.draw_button.config(
            text="Disable Pen" if self.drawing else "Enable Pen"
        )

    def choose_color(self):
        rgb, _ = colorchooser.askcolor(title="Choose mask color")
        if rgb:
            self.mask_color = tuple(int(c) for c in rgb)

    def handle_mouse(self, event):
        x, y = event.x, event.y
        if self.erasing:
            cv2.circle(self.binary_mask, (x, y), self.brush_size, 0, -1)
        elif self.drawing:
            cv2.circle(self.binary_mask, (x, y), self.brush_size, 255, -1)
        self.update_overlay()

    def save_mask(self):
        os.makedirs("mask", exist_ok=True)
        save_path = os.path.join("mask", "final_mask.png")
        cv2.imwrite(save_path, self.binary_mask)
        print(f"Mask 已儲存為 {save_path}")

# 測試用
if __name__ == "__main__":
    normal = r"D:\Git\robotlearning\mask_make\pass_image\Image_20250630155531861_aug_3.png"
    defect = r"D:\Git\robotlearning\mask_make\target\Image_20250701111128524_aug_2.png"
    MaskEditor(normal, defect)
