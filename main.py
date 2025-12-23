from ultralytics import YOLO
import cv2

cap = cv2.VideoCapture('vehicles.mp4')
model = YOLO('yolo11x.pt')
class_name = model.names

# linein & out

limitIn = [600, 1330, 1700, 1330]  
limitOut = [2250, 1300, 3500, 1300]  

# ONLY CHANGE (list → set)
totalCountIn = set()
totalCountOut = set()

classCountIn = {'car': 0, 'bus': 0, 'truck': 0}
classCountOut = {'car': 0, 'bus': 0, 'truck': 0}

# ===== Video Writer (1280x720) =====
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(
    'output_1280x7200.mp4',
    fourcc,
    20,
    (1280, 720)
)
# ==================================

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(
        frame,
        persist=True,
        tracker="botsort.yaml",
        conf=0.4,
        verbose=False
    )

    cv2.line(frame, (limitIn[0], limitIn[1]), (limitIn[2], limitIn[3]), (0,0,255), 3)
    cv2.line(frame, (limitOut[0], limitOut[1]), (limitOut[2], limitOut[3]), (0,0,255), 3)

    for i in results:
        if i.boxes is None:
            continue

        for box in i.boxes:
            if box.id is None:
                continue

            x1,y1,x2,y2 = box.xyxy[0]
            x1,y1,x2,y2 = int(x1), int(y1), int(x2), int(y2)
            cls = int(box.cls[0])
            name = class_name[cls]
            id = int(box.id[0])

            if name not in ['car', 'bus', 'truck']:
                continue

            cx, cy = (x1 + x2)//2, (y1 + y2)//2

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, f'{id} {name}',
                        (max(0,x1), max(40,y1)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        2,(255,0,255),3)
            cv2.circle(frame, (cx, cy), 5, (255,0,0), -1)

            # -------- IN --------
            if limitIn[0] < cx < limitIn[2] and limitIn[1]-15 < cy < limitIn[1]+15:
                if id not in totalCountIn:
                    totalCountIn.add(id)
                    classCountIn[name] += 1
                    cv2.line(frame, (limitIn[0], limitIn[1]),
                             (limitIn[2], limitIn[3]), (0,255,0), 3)

            # -------- OUT --------
            if limitOut[0] < cx < limitOut[2] and limitOut[1]-15 < cy < limitOut[1]+15:
                if id not in totalCountOut:
                    totalCountOut.add(id)
                    classCountOut[name] += 1
                    cv2.line(frame, (limitOut[0], limitOut[1]),
                             (limitOut[2], limitOut[3]), (0,255,0), 3)

    # ===== TEXT =====
    y0 = 60
    for cls in ['car', 'bus', 'truck']:
        text = f"{cls.upper()}  IN:{classCountIn[cls]}  OUT:{classCountOut[cls]}"
        cv2.putText(frame, text, (40, y0),
                    cv2.FONT_HERSHEY_DUPLEX,
                    2,
                    (0,255,255),
                    3)
        y0 += 60

    # ===== RESIZE & SAVE =====
    frame_720p = cv2.resize(frame, (1280, 720))
    out.write(frame_720p)

cap.release()
out.release()
cv2.destroyAllWindows()


print('saved video')
