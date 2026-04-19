from pynput.mouse import Controller, Button

mouse = Controller()


def left_click():
    mouse.click(Button.left, 1)


def right_click():
    mouse.click(Button.right, 1)


def double_click():
    mouse.click(Button.left, 2)


def drag_start():
    mouse.press(Button.left)


def drag_end():
    mouse.release(Button.left)


def scroll(dy):
    mouse.scroll(0, dy)
