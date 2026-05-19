import cv2
import pickle
import numpy as np

# ================= CONFIG =================
url = "http://192.0.0.4:8080/video"
file_name = "CarParkPos"
# ===========================================

# Try loading existing positions
try:
    with open(file_name, 'rb') as f:
        posList = pickle.load(f)
except:
    posList = []

drawing = False
start_point = None
current_mouse_pos = None


# ================= NIGHT ENHANCEMENT =================
def enhanceNight(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
# =====================================================


def mouseClick(event, x, y, flags, param):
    global drawing, start_point, posList, current_mouse_pos

    current_mouse_pos = (x, y)

    # LEFT CLICK
    if event == cv2.EVENT_LBUTTONDOWN:
        if not drawing:
            start_point = (x, y)
            drawing = True
        else:
            x1, y1 = start_point
            x2, y2 = x, y

            rect = (
                min(x1, x2),
                min(y1, y2),
                abs(x2 - x1),
                abs(y2 - y1)
            )

            if rect[2] > 20 and rect[3] > 20:  # prevent tiny boxes
                posList.append(rect)

                with open(file_name, 'wb') as f:
                    pickle.dump(posList, f)

                print(f"Slot Added: {rect}")

            drawing = False
            start_point = None

    # RIGHT CLICK (Delete)
    if event == cv2.EVENT_RBUTTONDOWN:
        for i, (px, py, pw, ph) in enumerate(posList):
            if px < x < px + pw and py < y < py + ph:
                posList.pop(i)
                with open(file_name, 'wb') as f:
                    pickle.dump(posList, f)
                print("Slot Removed")
                break


# Camera
cap = cv2.VideoCapture(url)

window_name = "DRAW PARKING SLOTS"
cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, mouseClick)

print("=== PARKING SLOT PICKER ===")
print("Left Click 2 Points → Create Slot")
print("Right Click → Delete Slot")
print("Press Q or ESC to Quit")

while True:
    success, img = cap.read()

    if not success:
        print("Camera disconnected. Retrying...")
        cap.release()
        cap = cv2.VideoCapture(url)
        continue

    # Apply night enhancement for accurate drawing
    img = enhanceNight(img)

    # Draw saved rectangles
    for (x, y, w, h) in posList:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)

    # Draw preview rectangle while dragging
    if drawing and start_point and current_mouse_pos:
        x1, y1 = start_point
        x2, y2 = current_mouse_pos
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 1)

    # Instruction overlay
    cv2.putText(img, f"Slots: {len(posList)}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2)

    cv2.imshow(window_name, img)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break

cap.release()
cv2.destroyAllWindows()