"""
face_iff.py
===========
Face-based Friend/Foe identification module.
Plugs into nuc_vision_bridge.py pipeline.

Folder structure:
    ~/turret/faces/
        gauty/
            img1.jpg
        friend2/
            img1.jpg

Install:
    pip install face-recognition
"""

import os
import cv2
import numpy as np
import face_recognition
from pathlib import Path


class FaceIFF:
    def __init__(self, faces_dir: str, tolerance: float = 0.5):
        self.tolerance = tolerance
        self.known_names = []
        self.known_encodings = []
        self._load_faces(faces_dir)

    def _load_faces(self, faces_dir: str):
        faces_path = Path(faces_dir)
        if not faces_path.exists():
            print(f"[IFF] WARNING: faces directory '{faces_dir}' not found.")
            return

        total = 0
        for person_dir in sorted(faces_path.iterdir()):
            if not person_dir.is_dir():
                continue
            name = person_dir.name
            count = 0
            for img_path in list(person_dir.glob("*.jpg")) + list(person_dir.glob("*.png")):
                img = face_recognition.load_image_file(str(img_path))
                encodings = face_recognition.face_encodings(img)
                if encodings:
                    self.known_encodings.append(encodings[0])
                    self.known_names.append(name)
                    count += 1
            print(f"[IFF] Loaded {count} encodings for '{name}'")
            total += count
        print(f"[IFF] Total: {total} encodings, {len(set(self.known_names))} people")

    def add_person(self, name: str, image_path: str):
        img = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(img)
        if encodings:
            self.known_encodings.append(encodings[0])
            self.known_names.append(name)
            print(f"[IFF] Added '{name}' from {image_path}")
            return True
        print(f"[IFF] No face found in {image_path}")
        return False

    def identify(self, frame: np.ndarray, bbox: tuple) -> dict:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        pad = 20
        h, w = frame.shape[:2]
        crop = frame[max(0,y1-pad):min(h,y2+pad), max(0,x1-pad):min(w,x2+pad)]
        if crop.size == 0:
            return {"name": "unknown", "status": "FOE", "conf": 0.0}

        rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        small    = cv2.resize(rgb_crop, (0,0), fx=0.5, fy=0.5)

        face_locs = face_recognition.face_locations(small, model="hog")
        if not face_locs:
            return {"name": "unknown", "status": "FOE", "conf": 0.0}

        encodings = face_recognition.face_encodings(small, face_locs)
        if not encodings:
            return {"name": "unknown", "status": "FOE", "conf": 0.0}

        face_enc = encodings[0]
        if not self.known_encodings:
            return {"name": "unknown", "status": "FOE", "conf": 0.0}

        distances = face_recognition.face_distance(self.known_encodings, face_enc)
        best_idx  = int(np.argmin(distances))
        best_dist = float(distances[best_idx])
        confidence = max(0.0, 1.0 - best_dist)

        if best_dist <= self.tolerance:
            return {"name": self.known_names[best_idx], "status": "FRIEND", "conf": confidence}
        return {"name": "unknown", "status": "FOE", "conf": confidence}

    def enroll_from_frame(self, frame: np.ndarray, bbox: tuple,
                          name: str, save_dir: str) -> bool:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        crop = frame[y1:y2, x1:x2]
        save_path = Path(save_dir) / name
        save_path.mkdir(parents=True, exist_ok=True)
        existing = list(save_path.glob("*.jpg"))
        out_path = save_path / f"img{len(existing):03d}.jpg"
        cv2.imwrite(str(out_path), crop)
        return self.add_person(name, str(out_path))