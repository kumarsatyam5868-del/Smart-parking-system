import cv2
import pickle
import numpy as np
import cvzone

# ================= CONFIG =================
url = "http://192.0.0.4:8080/video"

OCCUPANCY_THRESHOLD = 0.10     # motion ratio threshold
EDGE_THRESHOLD = 0.07          # structural edge ratio threshold
STABILITY_FRAMES = 8           # frames required to confirm change
LEARNING_RATE = 0.001          # slow background learning
# ===========================================


# Load parking positions
with open('CarParkPos', 'rb') as f:
    posList = pickle.load(f)

cap = cv2.VideoCapture(url)

# Background subtractor (KNN works better for parking lots)
fgbg = cv2.createBackgroundSubtractorKNN(detectShadows=True)

# Parking state memory
parking_state = {
    i: {"status": "FREE", "counter": 0}
    for i in range(len(posList))
}


# ================= NIGHT ENHANCEMENT =================
def enhanceNight(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
# =====================================================


# ================= SLOT CHECKING =====================
def checkParkingSpace(fgmask, frame):
    freeSpaces = 0

    for idx, (x, y, w, h) in enumerate(posList):

        roi_mask = fgmask[y:y+h, x:x+w]
        total_pixels = w * h

        # --- Motion ratio ---
        motion_pixels = cv2.countNonZero(roi_mask)
        motion_ratio = motion_pixels / total_pixels

        # --- Structural edge ratio from original image ---
        roi_color = frame[y:y+h, x:x+w]
        gray_roi = cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray_roi, 50, 150)
        edge_pixels = cv2.countNonZero(edges)
        edge_ratio = edge_pixels / total_pixels

        # --- Combined decision ---
        detected_status = "FREE"

        if motion_ratio > OCCUPANCY_THRESHOLD or edge_ratio > EDGE_THRESHOLD:
            detected_status = "OCCUPIED"

        # --- Stability smoothing ---
        previous_status = parking_state[idx]["status"]
        counter = parking_state[idx]["counter"]

        if detected_status != previous_status:
            counter += 1
            if counter >= STABILITY_FRAMES:
                parking_state[idx]["status"] = detected_status
                parking_state[idx]["counter"] = 0
            else:
                parking_state[idx]["counter"] = counter
        else:
            parking_state[idx]["counter"] = 0

        final_status = parking_state[idx]["status"]

        # --- Visualization ---
        if final_status == "FREE":
            color = (0, 255, 0)
            thickness = 2
            freeSpaces += 1
        else:
            color = (0, 0, 255)
            thickness = 3

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, thickness)

        cvzone.putTextRect(
            frame,
            final_status,
            (x, y+h-10),
            scale=0.6,
            thickness=1,
            offset=3,
            colorR=color
        )

    # Dashboard display
    cvzone.putTextRect(
        frame,
        f'AI PARKING SYSTEM: {freeSpaces}/{len(posList)} FREE',
        (40, 50),
        scale=1.2,
        thickness=2,
        offset=15,
        colorR=(30, 30, 30)
    )
# =====================================================


print("System Starting...")
print("Keep parking empty for 10 seconds for calibration.")
print("Press Q to quit.")

while True:
    success, img = cap.read()

    if not success:
        print("Camera connection lost. Retrying...")
        cap.release()
        cap = cv2.VideoCapture(url)
        continue

    # Enhance low light
    img = enhanceNight(img)

    # Blur for noise reduction
    blur = cv2.GaussianBlur(img, (5, 5), 3)

    # Background subtraction
    fgmask = fgbg.apply(blur, learningRate=LEARNING_RATE)

    # Remove shadows (gray pixels = 127)
    _, fgmask = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)

    # Morphology cleaning
    kernel = np.ones((5, 5), np.uint8)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
    fgmask = cv2.dilate(fgmask, kernel, iterations=1)

    # Check parking slots
    checkParkingSpace(fgmask, img)

    cv2.imshow("Smart Parking System - Team Satyam", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()