import cv2
import numpy as np
import os

# -------------------------------
# SETTINGS
# -------------------------------
KNOWN_FACES_DIR = "known_faces"
CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
SIFT = cv2.SIFT_create()

# Load face detector
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

# -------------------------------
# CHECK FOLDER
# -------------------------------
if not os.path.exists(KNOWN_FACES_DIR):
    print("❌ Folder 'known_faces' not found. Please create it.")
    exit()

# -------------------------------
# LOAD KNOWN FACES
# -------------------------------
known_faces = {}
known_names = []

for file in os.listdir(KNOWN_FACES_DIR):
    try:
        img_path = os.path.join(KNOWN_FACES_DIR, file)
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"⚠️ Could not load image: {file}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) > 0:
            name = os.path.splitext(file)[0]
            # Extract face region and compute SIFT keypoints
            x, y, w, h = faces[0]
            face_roi = gray[y:y+h, x:x+w]
            kp, des = SIFT.detectAndCompute(face_roi, None)
            
            if des is not None:
                known_faces[name] = des
                known_names.append(name)
                print(f"✅ Loaded face: {name}")
            else:
                print(f"⚠️ Could not extract features from {file}")
        else:
            print(f"⚠️ No face found in {file}, skipped.")

    except Exception as e:
        print(f"Error loading {file}: {e}")

print(f"✅ Total Faces Loaded: {len(known_names)}")

# Create BFMatcher for feature matching
FLANN = cv2.FlannBasedMatcher({'algorithm': 6, 'table_number': 12, 'key_size': 20, 'multi_probe_level': 2}, {})

# -------------------------------
# AUTO CAMERA DETECTION 
# -------------------------------
def get_working_camera():
    """Try multiple camera indices without specific backends"""
    for index in range(10):
        print(f"Trying camera index {index}...", end=" ")
        cap = cv2.VideoCapture(index)
        
        if cap.isOpened():
            # Test if camera actually works
            ret, frame = cap.read()
            if ret and frame is not None:
                print("✅ Found!")
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 30)
                return cap
            else:
                print("⚠️ Opened but not responding")
                cap.release()
        else:
            print("❌ Not available")
    
    return None

cap = get_working_camera()

if cap is None:
    print("\n" + "="*60)
    print("❌ NO CAMERA FOUND - PERMISSION/AVAILABILITY ISSUE")
    print("="*60)
    print("\n🔧 QUICK FIXES:")
    print("   1. Windows Settings > Privacy & Security > Camera")
    print("      → Enable camera for Python apps")
    print("\n   2. Device Manager")
    print("      → Check if camera is enabled (not disabled)")
    print("\n   3. Close other apps using camera (Zoom, Teams, etc.)")
    print("\n   4. Run as Administrator:")
    print("      → Right-click Command Prompt → Run as Administrator")
    print("      → Then run: python main.py")
    print("\n   5. Restart your computer")
    print("="*60)
    exit()

# Set camera properties for better performance
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

# -------------------------------
# MAIN LOOP
# -------------------------------
while True:
    ret, frame = cap.read()

    if not ret:
        print("⚠️ Failed to grab frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in current frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(50, 50))

    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        kp, des = SIFT.detectAndCompute(face_roi, None)
        
        name = "Unknown"
        
        # Match with known faces
        if des is not None and len(known_faces) > 0:
            matches_count = {}
            
            for known_name, known_des in known_faces.items():
                try:
                    knn_matches = FLANN.knnMatch(des, known_des, k=2)
                    
                    # Lowe's ratio test
                    good_matches = 0
                    for match_pair in knn_matches:
                        if len(match_pair) == 2:
                            m, n = match_pair
                            if m.distance < 0.75 * n.distance:
                                good_matches += 1
                    
                    matches_count[known_name] = good_matches
                except:
                    pass
            
            if matches_count:
                best_match = max(matches_count, key=matches_count.get)
                if matches_count[best_match] > 10:  # Minimum good matches
                    name = best_match

        # Draw rectangle
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Label
        cv2.putText(frame, name, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Face Recognition AI", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# -------------------------------
# CLEANUP
# -------------------------------
cap.release()
cv2.destroyAllWindows()