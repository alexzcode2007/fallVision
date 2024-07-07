import cv2
import supervision as sv
from ultralytics import YOLO
from modelanalytics import save_file

# Load the YOLOv8 model
model = YOLO('best5.pt')

annotator = sv.BoxAnnotator(
        thickness=4,
        text_thickness=2,
        text_scale=1
        )

# Open the video file
cap = cv2.VideoCapture(0)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True)
        detections = sv.Detections.from_ultralytics(results[0])

        labels = [
                f"{model.model.names[class_id]} {confidence:0.2f}"
                for _, _, confidence, class_id, *_ in detections
        ]

        for label in labels:
            if label.split()[0] == "fall":
                print("saving")
                save_file(frame)
                #print(labels)
                #with open("fall.log", "w") as file:
                #    file.write(label)

        # Visualize the results on the frame
        annotated_frame = annotator.annotate(
                scene=frame, detections=detections, labels=labels
        )

        # Display the annotated frame
        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
