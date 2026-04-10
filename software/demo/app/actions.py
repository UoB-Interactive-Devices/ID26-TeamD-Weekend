import time
from dataclasses import dataclass
from typing import Optional

import pyautogui

from .config import (
    ACCELERATION_RATE,
    BASE_MOVE_SPEED,
    CLICK_DELAY,
    MAX_MOVE_SPEED,
    MOUSE_MOVE_DELAY,
)

AVAILABLE_MACROS = {
    "move_left": "Move the mouse cursor to the left",
    "move_right": "Move the mouse cursor to the right",
    "move_up": "Move the mouse cursor up",
    "move_down": "Move the mouse cursor down",
    "click": "Perform a standard quick left mouse click",
    "click_and_hold": "Hold the left mouse button down whilst the gesture is active",
    "scroll_up": "Scroll the page up",
    "scroll_down": "Scroll the page down",
    "volume_up": "Increase the system volume",
    "volume_down": "Decrease the system volume",
    "mute": "Mute or unmute the system volume",
    "play_pause": "Play or pause current media",
    "copy": "Copy selected text or item",
    "paste": "Paste copied text or item",
    "undo": "Undo the last action",
    "switch_window": "Switch to the next open window",
    "zoom_in": "Zoom in on the active application",
    "zoom_out": "Zoom out on the active application",
}

DISCRETE_MACROS = {
    "click",
    "play_pause",
    "copy",
    "paste",
    "undo",
    "switch_window",
    "mute",
    "volume_up",
    "volume_down",
    "zoom_in",
    "zoom_out",
}

DEFAULT_GESTURE_MAPPING = {
    "left": "move_left",
    "up": "move_up",
    "right": "move_right",
    "down": "move_down",
    "squeeze": "click",
}


@dataclass
class ActionResult:
    action_taken: bool
    action_name: str
    click_position: Optional[tuple[int, int]] = None


class ActionController:
    def __init__(self) -> None:
        self.last_move_time = 0.0
        self.last_click_time = 0.0
        self.current_move_speed = BASE_MOVE_SPEED
        self.is_mouse_held = False

    def release_mouse(self) -> None:
        if self.is_mouse_held:
            pyautogui.mouseUp()
            self.is_mouse_held = False
        self.current_move_speed = BASE_MOVE_SPEED

    def apply_macro(self, macro: str) -> ActionResult:
        now = time.time()
        movement_actions = {
            "move_left": (-1, 0),
            "move_right": (1, 0),
            "move_up": (0, -1),
            "move_down": (0, 1),
        }

        if macro in movement_actions:
            if self.is_mouse_held:
                pyautogui.mouseUp()
                self.is_mouse_held = False
            if now - self.last_move_time >= MOUSE_MOVE_DELAY:
                self.current_move_speed = min(
                    self.current_move_speed + ACCELERATION_RATE,
                    MAX_MOVE_SPEED,
                )
                dx, dy = movement_actions[macro]
                pyautogui.move(
                    dx * self.current_move_speed, dy * self.current_move_speed
                )
                self.last_move_time = now
                return ActionResult(True, macro)
            return ActionResult(False, macro)

        self.current_move_speed = BASE_MOVE_SPEED

        if macro == "click_and_hold":
            if not self.is_mouse_held:
                pyautogui.mouseDown()
                self.is_mouse_held = True
            return ActionResult(True, macro)

        if self.is_mouse_held:
            pyautogui.mouseUp()
            self.is_mouse_held = False

        if macro in DISCRETE_MACROS and now - self.last_click_time >= CLICK_DELAY:
            click_position: Optional[tuple[int, int]] = None
            if macro == "click":
                pyautogui.click()
                current_pos = pyautogui.position()
                click_position = (current_pos.x, current_pos.y)
            elif macro == "scroll_up":
                pyautogui.scroll(200)
            elif macro == "scroll_down":
                pyautogui.scroll(-200)
            elif macro == "play_pause":
                pyautogui.press("playpause")
            elif macro == "copy":
                pyautogui.hotkey("ctrl", "c")
            elif macro == "paste":
                pyautogui.hotkey("ctrl", "v")
            elif macro == "undo":
                pyautogui.hotkey("ctrl", "z")
            elif macro == "switch_window":
                pyautogui.hotkey("alt", "tab")
            elif macro == "mute":
                pyautogui.press("volumemute")
            elif macro == "volume_up":
                pyautogui.press("volumeup")
            elif macro == "volume_down":
                pyautogui.press("volumedown")
            elif macro == "zoom_in":
                pyautogui.hotkey("ctrl", "+")
            elif macro == "zoom_out":
                pyautogui.hotkey("ctrl", "-")

            self.last_click_time = now
            return ActionResult(True, macro, click_position)

        return ActionResult(False, macro)
