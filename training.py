import streamlit as st
import numpy as np
from PIL import Image, ImageEnhance
import io
import rawpy
import cv2
import torch
from rembg import remove
import mediapipe as mp
from torchvision import transforms
import requests
from io import BytesIO
import logging
from streamlit_drawable_canvas import st_canvas

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize MediaPipe for face detection (for face swapping)
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# Placeholder for inpainting model (e.g., LaMa or Stable Diffusion)
def inpaint_image(image, mask):
    # Convert PIL image and mask to OpenCV format
    img_array = np.array(image)
    mask_array = np.array(mask)[:, :, 0]  # Convert mask to grayscale
    mask_array = cv2.resize(mask_array, (img_array.shape[1], img_array.shape[0]))
    
    # Simple inpainting using OpenCV (replace with LaMa or Stable Diffusion for realism)
    inpainted = cv2.inpaint(img_array, mask_array, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    return Image.fromarray(inpainted)

# Placeholder for face swapping
def face_swap(image, source_face_image):
    img_array = np.array(image)
    source_face_array = np.array(source_face_image)
    
    # Detect faces in the target image
    results = face_detection.process(cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
    if results.detections:
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            h, w = img_array.shape[:2]
            x, y = int(bbox.xmin * w), int(bbox.ymin * h)
            width, height = int(bbox.width * w), int(bbox.height * h)
            
            # Resize source face to match target face
            source_face_resized = cv2.resize(source_face_array, (width, height))
            
            # Simple blending (replace with DLib or DeepFace for better results)
            img_array[y:y+height, x:x+width] = source_face_resized
        return Image.fromarray(img_array)
    else:
        st.warning("No faces detected in the target image.")
        return image

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
        output = remove(img_array)
        return Image.fromarray(output)
    except Exception as e:
        logger.error(f"Error removing background: {str(e)}")
        raise

# Function for torso cropping using MediaPipe
def crop_torso(image):
    try:
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
        img_array = np.array(image)
        results = pose.process(cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            height, width = img_array.shape[:2]
            shoulder_y = min(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y,
                            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y) * height
            hip_y = max(landmarks[mp_pose.PoseLandmark.LEFT_HIP].y,
                        landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y) * height
            return image.crop((0, shoulder_y, width, hip_y))
        else:
            st.warning("No person detected. Using default crop.")
            return image.crop((0, height * 0.25, width, height * 0.75))
    except Exception as e:
        logger.error(f"Error cropping torso: {str(e)}")
        raise

# Streamlit app
def main():
    st.title("Advanced AI Image Manipulation App with Manual Editing")
    st.write("Upload an image or video and choose an AI-powered or manual editing effect.")

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
                ["None", "Remove Background", "Crop Torso", "Manual Cloth Removal", "Manual Face Swap", "Manual Object Removal"]
            )

            processed_image = image

            if manipulation_type != "None":
                # Manual editing canvas
                if manipulation_type in ["Manual Cloth Removal", "Manual Object Removal"]:
                    st.write("Draw on the image to select the area to edit (e.g., cloth or object).")
                    canvas_result = st_canvas(
                        fill_color="rgba(255, 165, 0, 0.3)",  # Orange fill for mask
                        stroke_width=3,
                        stroke_color="black",
                        background_image=image,
                        height=image.size[1],
                        width=image.size[0],
                        drawing_mode="freedraw",
                        key="canvas"
                    )
                    if canvas_result.image_data is not None:
                        mask = Image.fromarray(canvas_result.image_data)
                        with st.spinner("Processing manual edit..."):
                            processed_image = inpaint_image(image, mask)
                
                elif manipulation_type == "Manual Face Swap":
                    st.write("Upload a source face image for swapping.")
                    source_face_file = st.file_uploader("Choose source face image", type=["jpg", "jpeg", "png"])
                    if source_face_file:
                        source_face_image = Image.open(source_face_file).convert('RGB')
                        with st.spinner("Processing face swap..."):
                            processed_image = face_swap(image, source_face_image)
                
                elif manipulation_type == "Remove Background":
                    with st.spinner("Processing..."):
                        processed_image = remove_background(image)
                
                elif manipulation_type == "Crop Torso":
                    with st.spinner("Processing..."):
                        processed_image = crop_torso(image)
                
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
