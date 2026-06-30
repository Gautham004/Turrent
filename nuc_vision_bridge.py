"""
nuc_vision_bridge.py
====================
Runs on the NUC (dronedome).
Pipeline: DJI Action 5 Pro (USB webcam) → YOLOv8 → Kalman filter → PID → Serial → ESP32

Usage:
    python3 nuc_vision_bridge.py

Dependencies (already installed on dronedome):
    pip install ultralytics opencv-python pyserial filterpy numpy
"""

import cv2
import serial
import time
import numpy as np
from ultralytics import YOLO
from filterpy.kalman import KalmanFilter

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
SERIAL_PORT   = "/dev/ttyUSB0"
BAUD_RATE     = 115200
CAMERA_INDEX  = 0
FRAME_WIDTH   = 1280
FRAME_HEIGHT  = 720
YOLO_MODEL    = "yolov8n_openvino_model/"
CONF_THRESH   = 0.50
FIRE_THRESH   = 0.75
HEADLESS      = False 

KP_PAN, KI_PAN, KD_PAN   = 0.08, 0.001, 0.02
KP_TILT, KI_TILT, KD_TILT = 0.08, 0.001, 0.02

PAN_CENTER, TILT_CENTER  = 90, 90
PAN_MIN,  PAN_MAX        = 0, 180
TILT_MIN, TILT_MAX       = 45, 135


def make_kalman():
    kf = KalmanFilter(dim_x=4, dim_z=2)
    dt = 1 / 30.0
    kf.F = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]])
    kf.H = np.array([[1,0,0,0],[0,1,0,0]])
    kf.R *= 5
    kf.Q *= 1.0
    kf.P *= 100
    kf.x = np.array([[FRAME_WIDTH/2],[FRAME_HEIGHT/2],[0],[0]])
    return kf


class PID:
    def __init__(self, kp, ki, kd, out_min=-45, out_max=45):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.out_min, self.out_max = out_min, out_max
        self._integral = 0.0
        self._prev_err = 0.0
        self._prev_t   = time.time()

    def update(self, error):
        now = time.time()
        dt  = max(now - self._prev_t, 1e-6)
        self._integral += error * dt
        derivative      = (error - self._prev_err) / dt
        output = self.kp*error + self.ki*self._integral + self.kd*derivative
        self._prev_err = error
        self._prev_t   = now
        return float(np.clip(output, self.out_min, self.out_max))

    def reset(self):
        self._integral = 0.0
        self._prev_err = 0.0


def open_serial():
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
        time.sleep(2)
        print(f"[SERIAL] Connected on {SERIAL_PORT}")
        return ser
    except serial.SerialException as e:
        print(f"[SERIAL] ERROR: {e}")
        print("  → Try: sudo chmod 666 /dev/ttyUSB0")
        return None


def send_command(ser, pan, tilt, conf, fire=False):
    if ser is None:
        return
    packet = f"P{pan:.1f}T{tilt:.1f}C{conf:.2f}F{1 if fire else 0}\n"
    try:
        ser.write(packet.encode())
    except serial.SerialException:
        print("[SERIAL] Write failed")


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    if not cap.isOpened():
        print(f"[CAM] ERROR: Cannot open /dev/video{CAMERA_INDEX}")
        return

    model    = YOLO(YOLO_MODEL, task='detect')
    kf       = make_kalman()
    pid_pan  = PID(KP_PAN,  KI_PAN,  KD_PAN)
    pid_tilt = PID(KP_TILT, KI_TILT, KD_TILT)
    ser      = open_serial()

    pan_angle  = float(PAN_CENTER)
    tilt_angle = float(TILT_CENTER)
    frame_cx   = FRAME_WIDTH  / 2
    frame_cy   = FRAME_HEIGHT / 2
    prev_time  = time.time()
    no_target_frames = 0
    NO_TARGET_LIMIT  = 30

    print("[TURRET] Running. Press Q to quit.")

    while True:
        ret, frame = cap.read()
        
        if not ret:
            break

        now = time.time()
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now
        dt = 1.0 / max(fps, 1.0)
        kf.F[0,2] = dt
        kf.F[1,3] = dt

        results = model(frame, classes=[0], verbose=False)[0]
        boxes   = results.boxes
        best_box, best_conf = None, 0.0

        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                conf = float(box.conf[0])
                if conf > CONF_THRESH and conf > best_conf:
                    best_conf = conf
                    best_box  = box

        if best_box is not None:
            no_target_frames = 0
            x1,y1,x2,y2 = best_box.xyxy[0].tolist()
            target_x = (x1+x2)/2
            target_y = (y1+y2)/2

            kf.predict()
            kf.update(np.array([[target_x],[target_y]]))
            smooth_x = float(kf.x[0][0])
            smooth_y = float(kf.x[1][0])

            err_pan  = smooth_x - frame_cx
            err_tilt = smooth_y - frame_cy

            pan_angle  += pid_pan.update(err_pan)
            tilt_angle += pid_tilt.update(err_tilt)
            pan_angle   = float(np.clip(pan_angle,  PAN_MIN,  PAN_MAX))
            tilt_angle  = float(np.clip(tilt_angle, TILT_MIN, TILT_MAX))

            on_target = (abs(err_pan) < 20 and abs(err_tilt) < 20)
            fire      = on_target and (best_conf >= FIRE_THRESH)

            send_command(ser, pan_angle, tilt_angle, best_conf, fire)

            cv2.rectangle(frame, (int(x1),int(y1)), (int(x2),int(y2)), (0,255,0), 2)
            cv2.circle(frame, (int(smooth_x),int(smooth_y)), 5, (0,0,255), -1)
            cv2.putText(frame, f"CONF:{best_conf:.2f} PAN:{pan_angle:.1f} TILT:{tilt_angle:.1f}",
                        (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            if fire:
                cv2.putText(frame, "FIRING", (10,65),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
        else:
            no_target_frames += 1
            kf.predict()
            if no_target_frames >= NO_TARGET_LIMIT:
                pid_pan.reset()
                pid_tilt.reset()
                send_command(ser, PAN_CENTER, TILT_CENTER, 0.0)
            cv2.putText(frame, "NO TARGET", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        cv2.putText(frame, f"FPS:{fps:.1f}", (FRAME_WIDTH-120,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
        if not HEADLESS:
            cv2.imshow("Turret Vision", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    send_command(ser, PAN_CENTER, TILT_CENTER, 0.0)
    cap.release()
    cv2.destroyAllWindows()
    if ser:
        ser.close()
    print("[TURRET] Shutdown complete.")


if __name__ == "__main__":
    main()