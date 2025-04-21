# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import io
from typing import Any
import streamlit as st
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration

from ultralytics import YOLO
from ultralytics.utils import LOGGER
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.downloads import GITHUB_ASSETS_STEMS


class VideoProcessor(VideoTransformerBase):
    def __init__(self, yolo_model, conf, iou, selected_ind, enable_trk):
        self.yolo_model = yolo_model
        self.conf = conf
        self.iou = iou
        self.selected_ind = selected_ind
        self.enable_trk = enable_trk
        self.frame_count = 0

    def transform(self, frame):
        self.frame_count += 1
        img = frame.to_ndarray(format="bgr24")
        
        # Gửi frame này đến container của global state để có thể hiển thị
        if hasattr(st.session_state, 'current_frame'):
            st.session_state.current_frame = img.copy()
        
        # Chỉ xử lý mỗi frame thứ 3 để giảm tải CPU (có thể điều chỉnh)
        if self.frame_count % 3 == 0:
            if self.enable_trk == "Yes":
                results = self.yolo_model.track(
                    img, conf=self.conf, iou=self.iou, classes=self.selected_ind, persist=True
                )
            else:
                results = self.yolo_model(img, conf=self.conf, iou=self.iou, classes=self.selected_ind)
            
            annotated_frame = results[0].plot()  # Add annotations on frame
            
            # Lưu frame đã xử lý vào global state
            if hasattr(st.session_state, 'processed_frame'):
                st.session_state.processed_frame = annotated_frame
        
        return frame  # Trả về frame gốc cho WebRTC stream


class Inference:
    """
    A class to perform object detection, image classification, image segmentation and pose estimation inference.
    """

    def __init__(self, **kwargs: Any):
        """
        Initialize the Inference class, checking Streamlit requirements and setting up the model path.

        Args:
            **kwargs (Any): Additional keyword arguments for model configuration.
        """
        check_requirements("streamlit>=1.29.0")  # scope imports for faster ultralytics package load speeds
        check_requirements("streamlit-webrtc>=0.47.0")  # Add WebRTC requirement
        import streamlit as st

        self.st = st  # Reference to the Streamlit module
        self.source = None  # Video source selection (webcam or video file)
        self.enable_trk = "No"  # Flag to toggle object tracking
        self.conf = 0.25  # Confidence threshold for detection
        self.iou = 0.45  # Intersection-over-Union (IoU) threshold for non-maximum suppression
        self.org_frame = None  # Container for the original frame display
        self.ann_frame = None  # Container for the annotated frame display
        self.vid_file_name = None  # Video file name or webcam index
        self.selected_ind = []  # List of selected class indices for detection
        self.model = None  # YOLO model instance

        self.temp_dict = {"model": None, **kwargs}
        self.model_path = None  # Model file path
        if self.temp_dict["model"] is not None:
            self.model_path = self.temp_dict["model"]

        # Khởi tạo session state cho frames
        if 'current_frame' not in self.st.session_state:
            self.st.session_state.current_frame = None
        if 'processed_frame' not in self.st.session_state:
            self.st.session_state.processed_frame = None

        LOGGER.info(f"Ultralytics Solutions: ✅ {self.temp_dict}")

    def web_ui(self):
        """Sets up the Streamlit web interface with custom HTML elements."""
        menu_style_cfg = """<style>MainMenu {visibility: hidden;}</style>"""  # Hide main menu style

        # Main title of streamlit application
        main_title_cfg = """<div><h1 style="color:#FF64DA; text-align:center; font-size:40px; margin-top:-50px;
        font-family: 'Archivo', sans-serif; margin-bottom:20px;">Ultralytics YOLO Streamlit Application</h1></div>"""

        # Subtitle of streamlit application
        sub_title_cfg = """<div><h4 style="color:#042AFF; text-align:center; font-family: 'Archivo', sans-serif; 
        margin-top:-15px; margin-bottom:50px;">Experience real-time object detection on your webcam with the power 
        of Ultralytics YOLO! 🚀</h4></div>"""

        # Set html page configuration and append custom HTML
        self.st.set_page_config(page_title="Ultralytics Streamlit App", layout="wide")
        self.st.markdown(menu_style_cfg, unsafe_allow_html=True)
        self.st.markdown(main_title_cfg, unsafe_allow_html=True)
        self.st.markdown(sub_title_cfg, unsafe_allow_html=True)

    def sidebar(self):
        """Configure the Streamlit sidebar for model and inference settings."""
        with self.st.sidebar:  # Add Ultralytics LOGO
            logo = "https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Original.svg"
            self.st.image(logo, width=250)

        self.st.sidebar.title("User Configuration")  # Add elements to vertical setting menu
        self.source = self.st.sidebar.selectbox(
            "Video",
            ("webcam", "video"),
        )  # Add source selection dropdown
        self.enable_trk = self.st.sidebar.radio("Enable Tracking", ("Yes", "No"))  # Enable object tracking
        self.conf = float(
            self.st.sidebar.slider("Confidence Threshold", 0.0, 1.0, self.conf, 0.01)
        )  # Slider for confidence
        self.iou = float(self.st.sidebar.slider("IoU Threshold", 0.0, 1.0, self.iou, 0.01))  # Slider for NMS threshold

        col1, col2 = self.st.columns(2)  # Create two columns for displaying frames
        self.org_frame = col1.empty()  # Container for original frame
        self.ann_frame = col2.empty()  # Container for annotated frame

    def source_upload(self):
        """Handle video file uploads through the Streamlit interface."""
        self.vid_file_name = ""
        if self.source == "video":
            vid_file = self.st.sidebar.file_uploader("Upload Video File", type=["mp4", "mov", "avi", "mkv"])
            if vid_file is not None:
                g = io.BytesIO(vid_file.read())  # BytesIO Object
                with open("ultralytics.mp4", "wb") as out:  # Open temporary file as bytes
                    out.write(g.read())  # Read bytes into file
                self.vid_file_name = "ultralytics.mp4"
        elif self.source == "webcam":
            self.vid_file_name = 0  # Use webcam index 0

    def configure(self):
        """Configure the model and load selected classes for inference."""
        # Add dropdown menu for model selection
        available_models = [x.replace("yolo", "YOLO") for x in GITHUB_ASSETS_STEMS if x.startswith("yolo11")]
        if self.model_path:  # If user provided the custom model, insert model without suffix as *.pt is added later
            available_models.insert(0, self.model_path.split(".pt")[0])
        selected_model = self.st.sidebar.selectbox("Model", available_models)

        with self.st.spinner("Model is downloading..."):
            self.model = YOLO(f"{selected_model.lower()}.pt")  # Load the YOLO model
            class_names = list(self.model.names.values())  # Convert dictionary to list of class names
        self.st.success("Model loaded successfully!")

        # Multiselect box with class names and get indices of selected classes
        selected_classes = self.st.sidebar.multiselect("Classes", class_names, default=class_names[:3])
        self.selected_ind = [class_names.index(option) for option in selected_classes]

        if not isinstance(self.selected_ind, list):  # Ensure selected_options is a list
            self.selected_ind = list(self.selected_ind)

    def inference(self):
        """Perform real-time object detection inference on video or webcam feed."""
        self.web_ui()  # Initialize the web interface
        self.sidebar()  # Create the sidebar
        self.source_upload()  # Upload the video source
        self.configure()  # Configure the app

        # Định nghĩa cấu hình RTC để làm việc với webcam
        rtc_configuration = RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        )

        if self.source == "webcam":
            self.st.info("Click 'Start' để bắt đầu phân tích video từ webcam.")
            
            # Khởi tạo WebRTC streamer
            ctx = webrtc_streamer(
                key="yolo-detection",
                video_processor_factory=lambda: VideoProcessor(
                    self.model, self.conf, self.iou, self.selected_ind, self.enable_trk
                ),
                rtc_configuration=rtc_configuration,
                media_stream_constraints={"video": True, "audio": False},
            )
            
            # Hiển thị frames từ session state
            if ctx.state.playing:
                # Cập nhật frame mỗi 100ms
                self.st.info("Webcam đang hoạt động. Phân tích video đang diễn ra...")
                
                # Tạo placeholder để hiển thị frame
                frame_placeholder = self.st.empty()
                
                # Sử dụng experimental_rerun để cập nhật UI
                if self.st.session_state.current_frame is not None:
                    self.org_frame.image(self.st.session_state.current_frame, channels="BGR", caption="Original Frame")
                if self.st.session_state.processed_frame is not None:
                    self.ann_frame.image(self.st.session_state.processed_frame, channels="BGR", caption="Detected Objects")
                
                # Sử dụng st.experimental_rerun() để cập nhật UI liên tục
                self.st.experimental_rerun()
            
        elif self.source == "video" and self.vid_file_name:
            # Xử lý video file như cũ
            if self.st.sidebar.button("Start"):
                stop_button = self.st.button("Stop")  # Button to stop the inference
                cap = cv2.VideoCapture(self.vid_file_name)  # Capture the video
                if not cap.isOpened():
                    self.st.error("Could not open video source.")
                    return

                while cap.isOpened():
                    success, frame = cap.read()
                    if not success:
                        self.st.warning("Finished processing the video.")
                        break

                    # Process frame with model
                    if self.enable_trk == "Yes":
                        results = self.model.track(
                            frame, conf=self.conf, iou=self.iou, classes=self.selected_ind, persist=True
                        )
                    else:
                        results = self.model(frame, conf=self.conf, iou=self.iou, classes=self.selected_ind)

                    annotated_frame = results[0].plot()  # Add annotations on frame

                    if stop_button:
                        cap.release()  # Release the capture
                        self.st.stop()  # Stop streamlit app

                    self.org_frame.image(frame, channels="BGR")  # Display original frame
                    self.ann_frame.image(annotated_frame, channels="BGR")  # Display processed frame

                cap.release()  # Release the capture


if __name__ == "__main__":
    import sys  # Import the sys module for accessing command-line arguments

    # Check if a model name is provided as a command-line argument
    args = len(sys.argv)
    model = sys.argv[1] if args > 1 else None  # Assign first argument as the model name if provided
    # Create an instance of the Inference class and run inference
    Inference(model=model).inference()