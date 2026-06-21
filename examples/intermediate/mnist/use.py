import pickle
import tkinter as tk
import numpy as np

with open(input("Model filename (default: 'mnist_model.nn'): ") or "mnist_model.nn", 'rb') as f:
    model = pickle.load(f)['model']

GRID_SIZE = 28
CELL_SIZE = 20


class Draw28x28:
    def __init__(self, root):
        self.root = root
        self.root.title("28x28 MNIST Predictor")

        self.canvas = tk.Canvas(root, width=GRID_SIZE * CELL_SIZE, height=GRID_SIZE * CELL_SIZE, bg="black")
        self.canvas.pack()

        tk.Button(tk.Frame(root), text="Clear", command=self.clear).pack()

        self.image = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        self.rects = [[None] * GRID_SIZE for _ in range(GRID_SIZE)]
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                self.rects[y][x] = self.canvas.create_rectangle(
                    x * CELL_SIZE, y * CELL_SIZE, (x + 1) * CELL_SIZE, (y + 1) * CELL_SIZE,
                    outline="gray", fill="black",
                )

        self.canvas.bind("<Button-1>", self.draw)
        self.canvas.bind("<B1-Motion>", self.draw)

        self.label = tk.Label(root, text="Draw a digit", font=("Arial", 14), fg="white", bg="gray20")
        self.label.pack(pady=10)
        self.update_prediction()

    def draw(self, event):
        cx, cy = event.x // CELL_SIZE, event.y // CELL_SIZE
        kernel = {(0, 0): 1.0, (1, 0): 0.25, (-1, 0): 0.25, (0, 1): 0.25, (0, -1): 0.25,
                  (1, 1): 0.1, (-1, -1): 0.1, (1, -1): 0.1, (-1, 1): 0.1}
        for (dx, dy), v in kernel.items():
            x, y = cx + dx, cy + dy
            if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
                self.image[y][x] = min(1.0, self.image[y][x] + v * 0.35)
                self._update_cell(x, y)

    def _update_cell(self, x, y):
        intensity = int(self.image[y][x] * 255)
        color = f"#{intensity:02x}{intensity:02x}{intensity:02x}"
        self.canvas.itemconfig(self.rects[y][x], fill=color)

    def clear(self):
        self.image.fill(0)
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                self.canvas.itemconfig(self.rects[y][x], fill="black")
        self.label.config(text="Draw a digit")

    def update_prediction(self):
        x_b = self.image.flatten().reshape(1, -1)
        probs = model.forward(x_b)[0]
        digit = np.argmax(probs)
        conf = probs[digit] * 100
        self.label.config(text=f"Predicted: {digit} ({conf:.1f}%)")
        self.root.after(100, self.update_prediction)


if __name__ == "__main__":
    root = tk.Tk()
    Draw28x28(root)
    root.mainloop()
