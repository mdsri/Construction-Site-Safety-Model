import streamlit as st
from PIL import Image
import cv2
import numpy as np
from ultralytics import YOLO
import os
import re
import json
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# File path for storing reports
REPORTS_FILE = "reports.json"

# Load existing reports from file
def load_reports(file_path=REPORTS_FILE):
    try:
        with open(file_path, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        return []

# Save reports to file
def save_reports(reports, file_path=REPORTS_FILE):
    with open(file_path, "w") as file:
        json.dump(reports, file, indent=4)

# Function to load the YOLO model
def load_model(model_path='CSS_Model.pt'):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {os.path.abspath(model_path)}")
    return YOLO(model_path)

# Function to draw bounding boxes on the image
def draw_boxes(image, results, class_names):
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    for box in results[0].boxes:
        xyxy = box.xyxy[0].cpu().numpy()  # Get coordinates
        conf = box.conf[0].cpu().numpy()  # Confidence score
        class_id = int(box.cls[0].cpu().numpy())  # Class ID
        class_name = class_names.get(class_id, f"Class {class_id}")

        x1, y1, x2, y2 = map(int, xyxy)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Draw the box
        label = f"{class_name} {conf:.2f}"
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return img

# Function to validate email
def is_valid_email(email):
    return re.match(r"[^@]+@[^@]+\.[^@]+", email)

# Function to send email
def send_email(receiver_email, subject, body):
    sender_email = "FaultGuardAI@hotmail.com"
    sender_password = "Aa123123@"  # Replace with your actual password

    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = receiver_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP("smtp.office365.com", 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)
        st.success("Email sent successfully!")
    except Exception as e:
        st.error(f"Failed to send email: {e}")

# Main app
def main():
    st.title("Upload, Detect, and Save Reports")

    # Define class names
    class_names = {
        0: "Hardhat",
        1: "Mask",
        2: "Safety Vest",
        3: "NO-Hardhat",
        4: "NO-Mask",
        5: "NO-Safety Vest"
    }

    # Load YOLO model
    try:
        model = load_model()
        st.success("Model loaded successfully!")
    except FileNotFoundError as e:
        st.error(f"Error: {e}")
        return

    # Load existing reports
    reports = load_reports()

    # Email input
    email = st.text_input("Enter your email to receive detection results:")
    if not email or not is_valid_email(email):
        st.error("Please enter a valid email to proceed.")
        return

    # Upload image
    uploaded_file = st.file_uploader("Upload an image for detection:", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        # Load and display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Run detection
        results = model(np.array(image))
        if results and hasattr(results[0], 'boxes') and len(results[0].boxes) > 0:
            img_with_boxes = draw_boxes(image, results, class_names)
            img_pil = Image.fromarray(cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB))
            st.image(img_pil, caption="Image with Detections", use_container_width=True)

            # Safety equipment check
            required_items = {"Hardhat", "Mask", "Safety Vest"}
            violations = {"NO-Hardhat", "NO-Mask", "NO-Safety Vest"}
            detected_items = set()
            detected_violations = set()
            detected_objects = {}

            for box in results[0].boxes:
                class_id = int(box.cls[0].cpu().numpy())
                class_name = class_names.get(class_id, f"Class {class_id}")
                detected_objects[class_name] = detected_objects.get(class_name, 0) + 1

                # Check for required items
                if class_name in required_items:
                    detected_items.add(class_name)

                # Check for violations
                if class_name in violations:
                    detected_violations.add(class_name)

            st.subheader("Detection Results:")
            for class_name, count in detected_objects.items():
                st.write(f"- {class_name}: {count}")

            # Determine compliance status
            missing_items = required_items - detected_items
            safety_status = "Compliant" if not missing_items and not detected_violations else "Non-Compliant"

            if missing_items or detected_violations:
                st.error(f"Missing safety equipment: {', '.join(missing_items)}")
                st.error(f"Detected violations: {', '.join(detected_violations)}")
            else:
                st.success("All required safety equipment detected!")

            # Save report
            report = {
                "id": email,  # Using email as a unique ID
                "date_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "wearing": list(detected_items),
                "missing_items": list(missing_items),
                "violations": list(detected_violations),
                "status": safety_status
            }
            reports.append(report)
            save_reports(reports)
            st.success("Report saved successfully!")

            # Send email with results
            subject = "Your Detection Results"
            body = f"Safety Status: {safety_status}\n\n"
            body += "Detected Objects:\n" + "\n".join([f"- {class_name}: {count}" for class_name, count in detected_objects.items()])
            if missing_items:
                body += f"\n\nMissing Items: {', '.join(missing_items)}"
            if detected_violations:
                body += f"\n\nDetected Violations: {', '.join(detected_violations)}"
            send_email(email, subject, body)
        else:
            st.error("No objects detected. Please try another image.")

if __name__ == "__main__":
    main()
