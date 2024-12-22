import streamlit as st
from ultralytics import YOLO
import numpy as np
import cv2
from PIL import Image

# Define your class names (you may need to adjust this based on your trained model)
classes = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person',
           'Safety Cone', 'Safety Vest', 'machinery', 'vehicle']

# Function to load the YOLO model (using the ultralytics library)
def load_model(model_path='../Model/CSS_Model.pt'):
    model = YOLO(model_path)  # Load the YOLO model
    return model

# Function to draw bounding boxes on the image
def draw_boxes(image, results):
    """Draw bounding boxes on the image."""
    img = np.array(image)
    for box in results[0].boxes:
        # Get the bounding box coordinates, confidence, and class ID
        xyxy = box.xyxy[0].cpu().numpy()  # Get coordinates (x1, y1, x2, y2)
        conf = box.conf[0].cpu().numpy()  # Confidence score
        class_id = int(box.cls[0].cpu().numpy())  # Class ID
        class_name = model.names[class_id]  # Class name

        # Convert bounding box coordinates from normalized to pixel values
        x1, y1, x2, y2 = map(int, xyxy)
        
        # Draw the bounding box and label on the image
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Draw the box
        label = f"{class_name} {conf:.2f}"
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return img

# Function to run live video feed with object tracking
def run_live_detection():
    # Initialize webcam or video stream
    cap = cv2.VideoCapture(0)  # 0 for webcam, or provide path to video file

    # Create a window to display the video feed in Streamlit
    frame_placeholder = st.empty()

    # Initialize tracking variables
    trackers = []  # List of object trackers
    bboxes = []    # List to store bounding box coordinates
    tracking_objects = []  # To track object class names

    while st.session_state.live_stream_active:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to PIL Image for YOLO model processing
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Run inference using the YOLO model
        results = model(pil_image)  # Run the inference (this will be a list of Results objects)

        # Draw bounding boxes on the frame (this is only for detection purposes)
        img_with_boxes = draw_boxes(pil_image, results)

        # If no trackers are initialized, start new tracking for detected objects
        if len(trackers) == 0:
            for box in results[0].boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, xyxy)
                bboxes.append((x1, y1, x2, y2))
                tracking_objects.append(model.names[int(box.cls[0].cpu().numpy())])
                
                # Initialize a new tracker for the detected object
                tracker = cv2.TrackerCSRT_create()  # Now available after installing opencv-contrib-python
                tracker.init(frame, (x1, y1, x2 - x1, y2 - y1))  # Initialize tracker
                trackers.append(tracker)

        else:
            # Update the trackers for each frame
            for i, tracker in enumerate(trackers):
                ret, bbox = tracker.update(frame)
                if ret:
                    # If tracking successful, draw the updated box
                    x1, y1, x2, y2 = [int(coord) for coord in bbox]
                    cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    label = f"{tracking_objects[i]}"
                    cv2.putText(img_with_boxes, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Convert the frame with tracked boxes back to PIL image format for Streamlit display
        img_pil = Image.fromarray(cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB))

        # Display the current frame with detection and tracking
        frame_placeholder.image(img_pil, caption="Live PPE Detection", use_container_width=True)

    # Release the video capture object and close any OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

# Streamlit interface
st.title("PPE Detection with YOLO")

# Sidebar options
st.sidebar.title("Options")
image_option = st.sidebar.radio("Choose Input Type", ('Upload Image', 'Live Detection'))

# Load the model (assuming the model file is in the current directory)
model_path = '../Model/CSS_Model.pt'  # Path to your YOLO model
model = load_model(model_path)

# Initialize session state for live stream
if 'live_stream_active' not in st.session_state:
    st.session_state.live_stream_active = False  # Initialize live stream state

# Handle image upload or live detection
if image_option == 'Upload Image':
    st.sidebar.title("Upload Image")
    uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read and display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image.", use_container_width=True)

        # Run inference using the YOLO model
        results = model(image)  # Run the inference (this will be a list of Results objects)

        # Check if any objects are detected
        if len(results[0].boxes) > 0:
            # Draw bounding boxes on the image
            img_with_boxes = draw_boxes(image, results)

            # Convert the image back to a format that can be displayed by Streamlit
            img_pil = Image.fromarray(cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB))
            st.image(img_pil, caption="Image with Detections", use_container_width=True)

            # Display detection summary
            detected_objects = {}
            for box in results[0].boxes:
                class_id = int(box.cls[0].cpu().numpy())
                class_name = model.names[class_id]
                if class_name in detected_objects:
                    detected_objects[class_name] += 1
                else:
                    detected_objects[class_name] = 1

            st.subheader("Detection Results:")
            for class_name, count in detected_objects.items():
                st.write(f"- {class_name}: {count}")
        else:
            st.write("No objects detected.")

elif image_option == 'Live Detection':
    st.subheader("Live Detection Stream")
    
    # Toggle button to start/stop live stream
    if st.button("Start Recording" if not st.session_state.live_stream_active else "Stop Recording"):
        st.session_state.live_stream_active = not st.session_state.live_stream_active  # Toggle state
        st.rerun()  # Trigger a re-run to update the UI and button text

    # Run live detection if active
    if st.session_state.live_stream_active:
        st.text("Press 'Stop Recording' to end the stream.")
        run_live_detection()
    else:
        st.text("Live Detection is stopped. Press 'Start Recording' to start the stream.")
