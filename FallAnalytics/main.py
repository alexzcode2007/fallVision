import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov0n.yaml').load("best.pt")
#model = YOLO('yolov8n.pt')
# Open the video file
cap = cv2.VideoCapture(0)


#cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # print("contents\n")

        for result in results:
            for box in result.boxes.cpu().numpy():
                if box.cls == 1:
                    print("success")
        #         else:
        #             print("nofall")

        # Display the annotated frame
        cv2.imshow("YOLOv8 Tracking", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    else:
        # Break the loop if the end of the video is reached
        print("failed")
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
