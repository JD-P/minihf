import pyatspi
import pyautogui
import time
import math

class ATSPIDesktopTurtle:
    """Assistive Technology Service Provider Interface and PyAutoGUI based Linux desktop
    interaction tool for weave-agent with Logo Turtle like movement semantics."""
    def __init__(self):
        self.registry = pyatspi.Registry()
        self.desktop = self.registry.getDesktop(0)
        self.heading = 0  # Initial heading in degrees (0 degrees is to the right)

    def forward(self, distance=10):
        radians = math.radians(self.heading)
        dx = distance * math.cos(radians)
        dy = distance * math.sin(radians)
        pyautogui.moveRel(dx, dy)

    def backward(self, distance=10):
        radians = math.radians(self.heading)
        dx = -distance * math.cos(radians)
        dy = -distance * math.sin(radians)
        pyautogui.moveRel(dx, dy)

    def right(self, distance=10):
        radians = math.radians(self.heading + 90)
        dx = distance * math.cos(radians)
        dy = distance * math.sin(radians)
        pyautogui.moveRel(dx, dy)

    def left(self, distance=10):
        radians = math.radians(self.heading - 90)
        dx = distance * math.cos(radians)
        dy = distance * math.sin(radians)
        pyautogui.moveRel(dx, dy)

    def goto(self, x, y):
        pyautogui.moveTo(x, y)

    def setx(self, x):
        current_x, current_y = pyautogui.position()
        pyautogui.moveTo(x, current_y)

    def sety(self, y):
        current_x, current_y = pyautogui.position()
        pyautogui.moveTo(current_x, y)

    def setheading(self, angle):
        self.heading = angle

    def home(self):
        pyautogui.moveTo(0, 0)

    def speed(self, speed):
        pyautogui.PAUSE = 1 / speed

    def input_string(self, text):
        pyautogui.typewrite(text)

    def input_key_combination(self, keys):
        pyautogui.hotkey(*keys)

    def get_screen_elements(self):
        elements = []
        for app in self.desktop:
            for child in app:
                elements.append(child)
        return elements

    def get_keyboard_focusable_elements(self):
        focusable_elements = []
        for app in self.desktop:
            for child in app:
                if child.get_state().contains(pyatspi.STATE_FOCUSABLE):
                    focusable_elements.append(child)
        return focusable_elements

    def get_current_object_under_cursor(self):
        x, y = pyautogui.position()
        obj = self.registry.getAccessibleAtPoint(x, y, pyatspi.DESKTOP_COORDS)
        return obj

    def get_current_object_with_keyboard_focus(self):
        focus = self.registry.getFocus()
        return focus

    def scan(self):
        elements_info = []
        for app in self.desktop:
            for child in app:
                extents = child.queryComponent().getExtents(pyatspi.DESKTOP_COORDS)
                x, y, width, height = extents.x, extents.y, extents.width, extents.height
                elements_info.append({
                    'element': child,
                    'x': x,
                    'y': y,
                    'width': width,
                    'height': height
                })
        return elements_info

    def get_element_text(self, element):
        text_content = []

        def traverse(elem):
            if elem is not None:
                try:
                    text_interface = elem.queryText()
                    if text_interface:
                        text_content.append(text_interface.getText(0, -1))
                except NotImplementedError:
                    pass

                for child in elem:
                    traverse(child)

        traverse(element)
        return "\n".join(text_content)

# Example usage
if __name__ == "__main__":
    turtle = ATSPIDesktopTurtle()

    # Scan the screen and print elements with their coordinates
    elements_info = turtle.scan()
    for info in elements_info:
        print(f"Element: {info['element']}, Coordinates: ({info['x']}, {info['y']}), Size: ({info['width']}, {info['height']})")

        # Get the text content of the element
        text_content = turtle.get_element_text(info['element'])
        if text_content:
            print(f"Text Content: {text_content}")
        else:
            print("No text content available.")
