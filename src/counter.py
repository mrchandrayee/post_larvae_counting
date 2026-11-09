from pathlib import Path
import csv
import cv2
import numpy as np

WINDOW = "Post-larvae Counting | Q: quit | S: save frame"

def detect_larvae(frame, min_area=5, max_area=1800, s_min=35, v_max=245):
    """Segment brown/orange larval bodies against a light background.

    This is a classical-vision baseline. Tune HSV/area parameters for the
    camera, water clarity, illumination and magnification in the sample.
    """
    hsv=cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Brown/orange range, split around hue wrap not needed for this range.
    lower=np.array([3, s_min, 25], dtype=np.uint8)
    upper=np.array([38, 255, v_max], dtype=np.uint8)
    mask=cv2.inRange(hsv, lower, upper)
    # Remove isolated sensor noise while retaining small compact larvae.
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel,iterations=1)
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel,iterations=1)
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    detections=[]
    for c in contours:
        area=cv2.contourArea(c)
        if area < min_area or area > max_area: continue
        x,y,w,h=cv2.boundingRect(c)
        if w < 2 or h < 2: continue
        # Reject long fibers/strands; keep roughly compact or mildly elongated blobs.
        aspect=max(w/h,h/w)
        if aspect > 5.5: continue
        detections.append((x,y,w,h,float(area)))
    return mask,detections

def annotate(frame, detections, count=None):
    out=frame.copy()
    for x,y,w,h,area in detections:
        cv2.rectangle(out,(x,y),(x+w,y+h),(210,30,230),1)
        cv2.circle(out,(x+w//2,y+h//2),1,(0,255,255),-1)
    shown=len(detections) if count is None else count
    cv2.rectangle(out,(8,8),(230,48),(25,25,25),-1)
    cv2.putText(out,f"Post-larvae: {shown}",(16,35),cv2.FONT_HERSHEY_SIMPLEX,.7,(60,255,60),2)
    return out

def save_csv(rows, path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="",encoding="utf-8") as f:
        writer=csv.writer(f); writer.writerow(["frame","count","x","y","width","height","area_px"])
        writer.writerows(rows)

def _params(args):
    return args.min_area,args.max_area,args.s_min,args.v_max

def run_image(args):
    frame=cv2.imread(args.source)
    if frame is None: raise SystemExit(f"Cannot open image: {args.source}")
    mask,dets=detect_larvae(frame,*_params(args))
    annotated=annotate(frame,dets)
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    cv2.imwrite(str(out/"annotated_image.jpg"),annotated)
    cv2.imwrite(str(out/"segmentation_mask.png"),mask)
    rows=[[0,len(dets),x,y,w,h,a] for x,y,w,h,a in dets]
    save_csv(rows,out/"detections.csv")
    print(f"Estimated count: {len(dets)}")
    print(f"Saved annotated image, mask and CSV in: {out.resolve()}")
    cv2.imshow(WINDOW,annotated); cv2.imshow("Segmentation mask",mask)
    cv2.waitKey(0); cv2.destroyAllWindows()

def run_video(args):
    cap=cv2.VideoCapture(args.source)
    if not cap.isOpened(): raise SystemExit(f"Cannot open video: {args.source}")
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    fps=cap.get(cv2.CAP_PROP_FPS) or 25
    w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer=cv2.VideoWriter(str(out/"counted_video.mp4"),cv2.VideoWriter_fourcc(*"mp4v"),fps,(w,h))
    rows=[]; frame_no=0
    cv2.namedWindow(WINDOW,cv2.WINDOW_NORMAL)
    while True:
        ok,frame=cap.read()
        if not ok: break
        mask,dets=detect_larvae(frame,*_params(args))
        shown=annotate(frame,dets)
        writer.write(shown)
        rows.extend([[frame_no,len(dets),x,y,bw,bh,a] for x,y,bw,bh,a in dets])
        cv2.imshow(WINDOW,shown)
        cv2.imshow("Segmentation mask",mask)
        frame_no+=1
        key=cv2.waitKey(1)&0xFF
        if key==ord("q") or key==27: break
        if key==ord("s"): cv2.imwrite(str(out/f"frame_{frame_no:06d}.jpg"),shown)
    cap.release(); writer.release(); cv2.destroyAllWindows()
    save_csv(rows,out/"detections.csv")
    print(f"Processed {frame_no} frames. Results: {out.resolve()}")

def run_webcam(args):
    cap=cv2.VideoCapture(0)
    if not cap.isOpened(): raise SystemExit("Could not open webcam.")
    cv2.namedWindow(WINDOW,cv2.WINDOW_NORMAL)
    while True:
        ok,frame=cap.read()
        if not ok: break
        mask,dets=detect_larvae(frame,*_params(args))
        cv2.imshow(WINDOW,annotate(frame,dets))
        cv2.imshow("Segmentation mask",mask)
        key=cv2.waitKey(1)&0xFF
        if key==ord("q") or key==27: break
    cap.release(); cv2.destroyAllWindows()
