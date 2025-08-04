import streamlit as st
import numpy as np
from PIL import Image, ImageEnhance
import io
import rawpy
import cv2
import torch
from rembg import remove  # U^2-Net-based background removal
import mediapipe as mp
from torchvision import transforms
import requests
from io import BytesIO
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize MediaPipe for pose estimation
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

# Placeholder for style transfer model (simplified example using pre-trained model)
def apply_style_transfer(image, style_url):
    # Download a style image (e.g., from a URL)
    response = requests.get(style_url)
    style_image = Image.open(BytesIO(response.content)).convert('RGB')
    
    # Placeholder for style transfer (requires a pre-trained model like VGG19)
    # In practice, use a library like torch-vision or a custom model
    return image  # Replace with actual style transfer logic

# Function for super-resolution using a placeholder ESRGAN model
def super_resolution(image):
    # Convert PIL image to tensor
    transform = transforms.ToTensor()
    img_tensor = transform(image).unsqueeze(0)
    
    # Placeholder for ESRGAN model inference
    # In practice, load a pre-trained ESRGAN model using PyTorch
   
    return image  # Replace with actual super-resolution logic

# Function to process NEF (RAW) files
def process_nef(file):
    try:
        with rawpy.imread(file) as raw:
            rgb = raw.postprocess()
        return Image.fromarray(rgb)
    except Exception as e:
        logger.error(f"Error processing NEF file: {str(e)}")
        raise

# Function for background removal using U^2-Net
def remove_background(image):
    try:
        img_array = np.array(image)
        output = remove(img_array)  # U^2-Net-based background removal
        return Image.fromarray(output)
    except Exception as e:
        logger.error(f"Error removing background: {str(e)}")
        raise

# Function for torso cropping using MediaPipe
def crop_torso(image):
    try:
        img_array = np.array(image)
        results = pose.process(cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            height, width = img_array.shape[:2]
            
            # Get shoulder and hip landmarks
            shoulder_y = min(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y,
                           landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y) * height
            hip_y = max(landmarks[mp_pose.PoseLandmark.LEFT_HIP].y,
                        landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y) * height
            
            # Crop torso region
            return image.crop((0, shoulder_y, width, hip_y))
        else:
            st.warning("No person detected in the image. Using default crop.")
            return image.crop((0, height * 0.25, width, height * 0.75))
    except Exception as e:
        logger.error(f"Error cropping torso: {str(e)}")
        raise

# Streamlit app
def main():
    st.title("Advanced AI Image Manipulation App")
    st.write("Upload an image or video and choose from a variety of AI-powered effects.")

    # File uploader for images and videos
    uploaded_file = st.file_uploader(
        "Choose an image or video", type=["jpg", "jpeg", "png", "nef", "tiff", "bmp", "mp4"]
    )

    if uploaded_file is not None:
        try:
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            if file_extension in ['mp4']:
                st.warning("Video processing is experimental. Extracting first frame.")
                cap = cv2.VideoCapture(uploaded_file)
                ret, frame = cap.read()
                if ret:
                    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                else:
                    st.error("Failed to extract video frame.")
                    return
                cap.release()
            elif file_extension == 'nef':
                image = process_nef(uploaded_file)
            else:
                image = Image.open(uploaded_file).convert('RGB')
            
            # Display original image
            st.image(image, caption="Original Image", use_column_width=True)

            # Manipulation options
            manipulation_type = st.selectbox(
                "Select Manipulation Type",
                ["None", "Remove Background", "Crop Torso", "Style Transfer", "Super Resolution"]
            )

            # Additional inputs for specific effects
            style_url = None
            if manipulation_type == "Style Transfer":
                style_url = st.text_input("Enter URL of style image", "https://example.com/style.jpg")

            if manipulation_type != "None":
                # Perform selected manipulation
                with st.spinner("Processing..."):
                    if manipulation_type == "Remove Background":
                        processed_image = remove_background(image)
                    elif manipulation_type == "Crop Torso":
                        processed_image = crop_torso(image)
                    elif manipulation_type == "Style Transfer":
                        if style_url:
                            processed_image = apply_style_transfer(image, style_url)
                        else:
                            st.error("Please provide a valid style image URL.")
                            return
                    elif manipulation_type == "Super Resolution":
                        processed_image = super_resolution(image)
                
                # Display processed image
                st.image(processed_image, caption=f"{manipulation_type} Result", use_column_width=True)

                # Option to download processed image
                buffered = io.BytesIO()
                processed_image.save(buffered, format="PNG")
                st.download_button(
                    label="Download Processed Image",
                    data=buffered.getvalue(),
                    file_name="processed_image.png",
                    mime="image/png"
                )

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            logger.error(f"Error in main: {str(e)}")

if __name__ == "__main__":
    main()
