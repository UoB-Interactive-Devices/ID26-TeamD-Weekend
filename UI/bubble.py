import random

WIDTH = 800
HEIGHT = 400

class Bubble:
    def __init__(self, canvas):
        self.canvas = canvas
        self.r = random.randint(15, 50)

        x = random.randint(self.r, WIDTH - self.r)
        y = random.randint(self.r, HEIGHT - self.r)

        self.id = canvas.create_oval(
            x - self.r, y - self.r, x + self.r, y + self.r,
            fill="skyblue", outline="white", width=2
        )

        self.dx = random.uniform(-2, 2)
        self.dy = random.uniform(-2, 2)

    def move(self):
        self.canvas.move(self.id, self.dx, self.dy)
        x1, y1, x2, y2 = self.canvas.coords(self.id)

        if x1 <= 0 or x2 >= WIDTH:
            self.dx *= -1
        if y1 <= 0 or y2 >= HEIGHT:
            self.dy *= -1

    def contains(self, x, y):
        x1, y1, x2, y2 = self.canvas.coords(self.id)
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        r = (x2 - x1) / 2
        return (x - cx)**2 + (y - cy)**2 <= r**2