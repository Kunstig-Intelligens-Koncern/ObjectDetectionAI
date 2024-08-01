import torch
from transformers import pipeline
import cv2
from torchvision import models, transforms
from PIL import Image

# Load the SAM model
sam_model = models.segmentation.deeplabv3_resnet101(pretrained=True)
sam_model.eval()

# Load the NLP model for text processing
nlp = pipeline("fill-mask", model="bert-base-uncased")

# Function to preprocess video frames
def preprocess_frame(frame):
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(frame)
    input_batch = input_tensor.unsqueeze(0)
    return input_batch

# Function to segment objects in the frame
def segment_frame(frame):
    input_batch = preprocess_frame(frame)
    with torch.no_grad():
        output = sam_model(input_batch)['out'][0]
    return output.argmax(0).byte().cpu().numpy()

# Initialize video capture
cap = cv2.VideoCapture(0)

# Process each frame from the video feed
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Segment the frame
    segmented_frame = segment_frame(rgb_frame)
    
    # Display the segmented frame
    cv2.imshow('Segmented Frame', segmented_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
