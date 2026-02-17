import tkinter as tk

class UI:

    def __init__(self, root):
        self.root = root
        self.root.title("Weekend Wednesday")

        self.frame = tk.Frame(root)
        self.frame.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(
            self.frame,
            width=400,
            height=200,
            bg="white"
        )
        self.canvas.pack(padx=10, pady=10, fill="both", expand=True)

        self.text_display = tk.Text(
            self.frame,
            height=5,
            font=("Consolas", 12),
            wrap="word"
        )
        self.text_display.pack(padx=10, pady=(0, 10), fill="x")

        # Make text read-only
        self.text_display.config(state="disabled")

        cursor_points = [
            (0, 0),  # tip of cursor
            (0, 18),  # left edge downward
            (4, 14),  # inner notch
            (7, 22),  # tail left
            (10, 21),
            (7, 13),  # inner return
            (14, 13),  # right edge
        ]

        self.cursor = self.canvas.create_polygon(
            [coord for point in cursor_points for coord in point],
            fill="white",
            outline="black",
        )

    def move_cursor(self, xOffset, yOffset):
        self.canvas.move(self.cursor, xOffset, yOffset)

    def click(self):
        print("click") # TODO implement

    def type(self, char):
        self.text_display.config(state="normal")

        match char:
            case 0xE2 | 0x90 | 0x88: # backspace in Unicode
                if self.text_display.compare("end-1c", ">", "1.0"):
                    self.text_display.delete("end-2c", "end-1c")
            case _: # just add the character
                self.text_display.insert("end", char)

        self.text_display.see("end")  # auto-scroll
        self.text_display.config(state="disabled")


# ---- Create window ----
window = tk.Tk()
interface = UI(window)

# Example usage:
def demo_typing():
    for ch in "Hello, Tkinter!!":
        interface.type(ch)
    interface.type(0xE2)


# Call demo after window loads
window.after(500, demo_typing)


def on_key(event):
    if event.keysym == "Left":
        interface.move_cursor(-10, 0)
    elif event.keysym == "Right":
        interface.move_cursor(10, 0)
    elif event.keysym == "Up":
        interface.move_cursor(0, -10)
    elif event.keysym == "Down":
        interface.move_cursor(0, 10)
    elif event.char:
        interface.type(event.char)


window.bind("<Key>", on_key)


window.mainloop()