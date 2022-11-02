from yolo_mosaic import yolo_mosaic
import argparse
import cv2
import time

if __name__ == "__main__":
    # Parse CLI arguments
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--net_filename', type=str, default='yolov5s-face.onnx',
                            help="Filename of the pre-trained model file, default: yolov5s-face.onnx")
    arg_parser.add_argument(
        '--live_demo', action='store_true', help="Live face mosaic demo using webcam")
    arg_parser.add_argument("--video_input", type=str,
                            default='input.mov', help="Filename of the input video, default: input.mov")
    arg_parser.add_argument("--video_output", type=str,
                            default='output.mov', help="Filename of the input video, default: output.mov")
    arg_parser.add_argument('--conf_threshold', default=0.3,
                            type=float, help='Value of class confidence threshold, optional, default: 0.3')
    arg_parser.add_argument('--nms_threshold', default=0.5,
                            type=float, help='Value of NMS-IoU threshold, optional, default: 0.5')
    arg_parser.add_argument('--obj_threshold', default=0.3,
                            type=float, help='Value of object confidence threshold, optional, default: 0.3')

    args = arg_parser.parse_args()

    yolonet = yolo_mosaic(args.net_filename, conf_threshold=args.conf_threshold,
                          nms_threshold=args.nms_threshold, obj_threshold=args.obj_threshold)

    # Determine the input source
    if args.live_demo == True:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(args.video_input, cv2.CAP_FFMPEG)
        out = cv2.VideoWriter(args.video_output, cv2.VideoWriter_fourcc(*"mp4v"), int(cap.get(cv2.CAP_PROP_FPS)),
                              (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    # For performance evaluation
    prev_frame_time = 0
    new_frame_time = 0
    start_time = time.time()
    frame_count = 0
    total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Loop until the end of video
    while(cap.isOpened()):
        # Frame count and time count
        new_frame_time = time.time()
        frame_count += 1

        # Get frame
        ret, frame = cap.read()
        if ret == False:
            break
        dets = yolonet.detect(frame)
        srcimg = yolonet.post_process(frame, dets)

        # Calculate real-time FPS
        fps = 1/(new_frame_time-prev_frame_time)
        print("frame:", frame_count, "/", total_frame, ", fps =", fps)

        if args.live_demo == True:
            # Show real-time image and FPS
            cv2.putText(frame, str(fps), (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow("Face Mosaic Demo", frame)
        else:
            # Write current frame to file
            out.write(frame)

        prev_frame_time = new_frame_time

    print("Done. Time elapsed:", time.time()-start_time,
          ", average fps:", total_frame/(time.time()-start_time))

    cap.release()
    if args.live_demo == True:
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        out.release()
