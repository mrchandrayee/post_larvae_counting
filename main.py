import argparse
from src.counter import run_image, run_video, run_webcam

def main():
    p = argparse.ArgumentParser(description="Post-larvae image/video counting")
    p.add_argument("--source", help="Image/video path; omit to open webcam")
    p.add_argument("--mode", choices=["auto","image","video"], default="auto")
    p.add_argument("--min-area", type=int, default=5)
    p.add_argument("--max-area", type=int, default=1800)
    p.add_argument("--s-min", type=int, default=35)
    p.add_argument("--v-max", type=int, default=245)
    p.add_argument("--output", default="output")
    args = p.parse_args()
    if not args.source:
        run_webcam(args)
    else:
        mode=args.mode
        if mode=="auto":
            import cv2
            cap=cv2.VideoCapture(args.source)
            is_video=cap.isOpened()
            cap.release()
            mode="video" if is_video else "image"
        if mode=="image": run_image(args)
        else: run_video(args)

if __name__ == "__main__":
    main()
