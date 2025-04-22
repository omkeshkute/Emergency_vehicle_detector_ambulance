import cv2
import numpy as np
import pytesseract

try:
    from playsound import playsound
    sound_enabled = True
except:
    sound_enabled = False

pytesseract.pytesseract.tesseract_cmd = r"C:\Users\omkes\Downloads\EmergencyVehicleDetection_Advanced\Tesseract-OCR\tesseract.exe"

cap = cv2.VideoCapture("emergency.mp4")

if not cap.isOpened():
    print("❌ Video not opened. Check path or format.")
    exit()
else:
    print("✅ Video loaded successfully.")


lower_red1 = np.array([0, 100, 100])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([160, 100, 100])
upper_red2 = np.array([180, 255, 255])
lower_blue = np.array([100, 100, 100])
upper_blue = np.array([130, 255, 255])

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Red & Blue masks
    red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = red_mask1 | red_mask2
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

    red_pixels = cv2.countNonZero(red_mask)
    blue_pixels = cv2.countNonZero(blue_mask)

    print(f"[Frame {frame_count}] Red Pixels: {red_pixels}, Blue Pixels: {blue_pixels}")

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detected_text = pytesseract.image_to_string(gray).upper()
    print(f"[Frame {frame_count}] Detected Text: {detected_text.strip()}")

    if "AMBULANCE" in detected_text or red_pixels > 800 or blue_pixels > 800:
        print("✅ Emergency Detected!")
        cv2.putText(frame, "🚨 Emergency Vehicle Detected - Signal Turned GREEN", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if sound_enabled:
            playsound("siren.mp3", block=False)

    cv2.imshow("Traffic Monitoring", frame)
    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()