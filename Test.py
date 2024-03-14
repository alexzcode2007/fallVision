from ultralytics import YOLO

# Load a model
model = YOLO("yolov0n.yaml")  # build a new model from scratch
model = model.load("yolov8n.pt")  # load a pretrained model (recommended for training)


# Use the model
model.train(data="dataset.yaml", epochs=1, cache=True)  # train the model
