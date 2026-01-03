from neural_network_vectorized_multiple_models import load_network
import tkinter as tk
import numpy as np

model = load_network(input("Enter model filename (default: 'mnist_model.nn'): ") or "mnist_model.nn")

GRID_SIZE = 28
CELL_SIZE = 20


class Draw28x28Smooth:
    def __init__(self, root):
        self.root = root
        self.root.title("28x28 MNIST Live Predictor")

        self.canvas = tk.Canvas(
            root,
            width=GRID_SIZE * CELL_SIZE,
            height=GRID_SIZE * CELL_SIZE,
            bg="black"
        )
        self.canvas.pack()

        btn_frame = tk.Frame(root)
        btn_frame.pack()

        tk.Button(btn_frame, text="Clear", command=self.clear).pack(side=tk.LEFT)

        self.image = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)

        self.rects = [[None] * GRID_SIZE for _ in range(GRID_SIZE)]
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                self.rects[y][x] = self.canvas.create_rectangle(
                    x * CELL_SIZE,
                    y * CELL_SIZE,
                    (x + 1) * CELL_SIZE,
                    (y + 1) * CELL_SIZE,
                    outline="gray",
                    fill="black"
                )

        self.canvas.bind("<Button-1>", self.draw)
        self.canvas.bind("<B1-Motion>", self.draw)

        self.prediction_label = tk.Label(
            root,
            text="Draw a digit",
            font=("Arial", 14),
            fg="white",
            bg="gray20"
        )
        self.prediction_label.pack(pady=10)

        self.update_prediction()

    def draw(self, event):
        cx = event.x // CELL_SIZE
        cy = event.y // CELL_SIZE

        kernel = {
            (0, 0): 1.0,
            (1, 0): 0.25, (-1, 0): 0.25,
            (0, 1): 0.25, (0, -1): 0.25,
            (1, 1): 0.1, (-1, -1): 0.1,
            (1, -1): 0.1, (-1, 1): 0.1
        }

        for (dx, dy), value in kernel.items():
            x = cx + dx
            y = cy + dy
            if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
                self.image[y][x] = min(1.0, self.image[y][x] + value * 0.35)
                self.update_cell(x, y)

    def update_cell(self, x, y):
        intensity = int(self.image[y][x] * 255)
        color = f"#{intensity:02x}{intensity:02x}{intensity:02x}"
        self.canvas.itemconfig(self.rects[y][x], fill=color)

    def clear(self):
        self.image.fill(0)
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                self.canvas.itemconfig(self.rects[y][x], fill="black")
        self.prediction_label.config(text="Draw a digit")

    def update_prediction(self):
        x_b = self.image.flatten()
        x_b = np.array([x_b])
        prediction = model.calculate_output(x_b)[0]
        top_digit = np.argmax(prediction)
        confidence = prediction[top_digit] * 100
        self.prediction_label.config(
            text=f"Predicted: {top_digit} ({confidence:.1f}%)"
        )
        self.root.after(100, self.update_prediction)


if __name__ == "__main__":
    root = tk.Tk()
    app = Draw28x28Smooth(root)
    root.mainloop()
