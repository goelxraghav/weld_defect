import streamlit as st
from ultralytics import YOLO
from PIL import Image

# Using caching to load the model only once
@st.cache_resource
def load_model(model_path):
    """Loads the YOLO model from the specified path."""
    model = YOLO(model_path)
    return model

# --- Page Configuration ---
st.set_page_config(
    page_title="Weld Defect Detection",
    page_icon="🤖",
    layout="wide"
)

# --- Main App ---
st.title("Weld Defect Detection ")
st.write("Upload an image to detect potential weld defects.")

# --- Model Loading ---
model = load_model('best.pt')

# --- Image Uploader ---
uploaded_file = st.file_uploader(
    "Choose a weld image...", 
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Open the uploaded image
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=False)

    # Perform prediction when the button is clicked
    if st.button("Detect Defects"):
        with st.spinner("Detecting..."):
            # Run prediction
            results = model.predict(image)
            result = results[0]
            
            # Display annotated image
            annotated_image = result.plot() # Returns a BGR NumPy array
            rgb_annotated_image = annotated_image[..., ::-1] # Convert to RGB

            with col2:
                st.image(rgb_annotated_image, caption="Detection Result", use_container_width=False)

            # Display detection details
            if len(result.boxes) == 0:
                st.info("No defects were detected.")
            else:
                st.success(f"Detected {len(result.boxes)} potential defects.")
                for box in result.boxes:
                    class_name = result.names[int(box.cls[0])]
                    confidence = float(box.conf[0])
                    st.write(f"- **Defect:** {class_name}, **Confidence:** {confidence:.2f}")