import cv2

video_source = 'metal_pipe.mp4'

for i in range(10):
    x = 0
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print("Error opening video stream or file")

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    out_avi = cv2.VideoWriter(f"metal_pipe_out_{i + 1}.avi", cv2.VideoWriter_fourcc("M", "J", "P", "G"), 30, (frame_width, frame_height))

    # Read until video is completed
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()

        if ret:
            if x == i:
                # Write the frame to the output files
                out_avi.write(frame[..., ::-1])
                x = 0
            else:
                x += 1
        # Break the loop
        else:
            break

    cap.release()
    out_avi.release()
