# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import io
from typing import Any

import cv2

from ultralytics import YOLO
from ultralytics.utils import LOGGER
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.downloads import GITHUB_ASSETS_STEMS


class Inference:
    """
    A class to perform object detection, image classification, image segmentation and pose estimation inference.

    This class provides functionalities for loading models, configuring settings, uploading video files, and performing
    real-time inference using Streamlit and Ultralytics YOLO models.

    Attributes:
        st (module): Streamlit module for UI creation.
        temp_dict (dict): Temporary dictionary to store the model path and other configuration.
        model_path (str): Path to the loaded model.
        model (YOLO): The YOLO model instance.
        source (str): Selected video source (webcam or video file).
        enable_trk (str): Enable tracking option ("Yes" or "No").
        conf (float): Confidence threshold for detection.
        iou (float): IoU threshold for non-maximum suppression.
        org_frame (Any): Container for the original frame to be displayed.
        ann_frame (Any): Container for the annotated frame to be displayed.
        vid_file_name (str | int): Name of the uploaded video file or webcam index.
        selected_ind (List[int]): List of selected class indices for detection.

    Methods:
        web_ui: Sets up the Streamlit web interface with custom HTML elements.
        sidebar: Configures the Streamlit sidebar for model and inference settings.
        source_upload: Handles video file uploads through the Streamlit interface.
        configure: Configures the model and loads selected classes for inference.
        inference: Performs real-time object detection inference.

    Examples:
        >>> inf = Inference(model="path/to/model.pt")  # Model is an optional argument
        >>> inf.inference()
    """

    def __init__(self, **kwargs: Any):
        """
        Initialize the Inference class, checking Streamlit requirements and setting up the model path.

        Args:
            **kwargs (Any): Additional keyword arguments for model configuration.
        """
        check_requirements("streamlit>=1.29.0")  # scope imports for faster ultralytics package load speeds
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
            
        return class_names

    def inference(self):
        """Perform real-time object detection inference on video or webcam feed."""
        self.web_ui()  # Initialize the web interface
        self.sidebar()  # Create the sidebar
        self.source_upload()  # Upload the video source
        class_names = self.configure()  # Configure the app
        
        # Tạo hai cột cho hiển thị kết quả
        col1, col2 = self.st.columns(2)
        with col1:
            self.st.write("Original Stream")
            self.org_frame = self.st.empty()
        with col2:
            self.st.write("Processed Stream")
            self.ann_frame = self.st.empty()
        
        # Xử lý trường hợp video
        if self.source == "video" and self.vid_file_name:
            if self.st.sidebar.button("Start"):
                stop_button = self.st.button("Stop")
                cap = cv2.VideoCapture(self.vid_file_name)
                if not cap.isOpened():
                    self.st.error("Could not open video source.")
                    return
                
                while cap.isOpened() and not stop_button:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Process with YOLO
                    if self.enable_trk == "Yes":
                        results = self.model.track(
                            frame, conf=self.conf, iou=self.iou, classes=self.selected_ind, persist=True
                        )
                    else:
                        results = self.model(frame, conf=self.conf, iou=self.iou, classes=self.selected_ind)
                    
                    # Convert BGR to RGB for Streamlit display
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # Plot results on frame
                    annotated_frame = results[0].plot()
                    annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    
                    # Display frames
                    self.org_frame.image(rgb_frame, channels="RGB")
                    self.ann_frame.image(annotated_frame_rgb, channels="RGB")
                    
                    # Check if stop button was pressed
                    if stop_button:
                        break
                
                cap.release()
        
        # Xử lý trường hợp webcam với streamlit-webrtc
        elif self.source == "webcam":
            check_requirements("streamlit-webrtc>=0.45.0 av>=10.0.0")
            from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
            import av
            
            class VideoProcessor(VideoProcessorBase):
                def __init__(self, model, conf, iou, selected_ind, enable_trk, class_names):
                    self.model = model
                    self.conf = conf
                    self.iou = iou
                    self.selected_ind = selected_ind
                    self.enable_trk = enable_trk
                    self.class_names = class_names
                    self.original_frame = None
                    
                def recv(self, frame):
                    img = frame.to_ndarray(format="bgr24")
                    self.original_frame = img.copy()
                    
                    # Process frame with model
                    if self.enable_trk == "Yes":
                        results = self.model.track(
                            img, conf=self.conf, iou=self.iou, classes=self.selected_ind, persist=True
                        )
                    else:
                        results = self.model(img, conf=self.conf, iou=self.iou, classes=self.selected_ind)
                    
                    # Annotate the frame with detection results
                    annotated_frame = results[0].plot()
                    
                    return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")
            
            # Configure WebRTC
            # Thay đổi phần cấu hình WebRTC trong phương thức inference()
            rtc_configuration = RTCConfiguration({
                    "iceServers": [
                        {
                            "urls": "stun:global.stun.twilio.com:3478"
                        },
                        {
                            "urls": "turn:global.turn.twilio.com:3478?transport=udp",
                            "username": "9d4853635b24303fed5bc727b3affd45b5a7e18723896e3a483a7079b4146317",
                            "credential": "+R3jJHFzw+LQgnxIDWjq+nx89MD4CUCMy+oDWwo63qc="
                        },
                        {
                            "urls": "turn:global.turn.twilio.com:3478?transport=tcp",
                            "username": "9d4853635b24303fed5bc727b3affd45b5a7e18723896e3a483a7079b4146317",
                            "credential": "+R3jJHFzw+LQgnxIDWjq+nx89MD4CUCMy+oDWwo63qc="
                        },
                        {
                            "urls": "turn:global.turn.twilio.com:443?transport=tcp",
                            "username": "9d4853635b24303fed5bc727b3affd45b5a7e18723896e3a483a7079b4146317",
                            "credential": "+R3jJHFzw+LQgnxIDWjq+nx89MD4CUCMy+oDWwo63qc="
                        }
                    ]
                }
            )
            
            # Start WebRTC streamer
            webrtc_ctx = webrtc_streamer(
                key="ultralytics-detection",
                video_processor_factory=lambda: VideoProcessor(
                    self.model, self.conf, self.iou, self.selected_ind, self.enable_trk, class_names
                ),
                rtc_configuration=rtc_configuration,
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,
            )
            
            # Information about WebRTC
            self.st.info("""
            ** Lưu ý về Webcam:**
            - Thông qua WebRTC, ứng dụng sẽ cần quyền truy cập vào webcam của bạn
            - Xử lý video được thực hiện trong trình duyệt của bạn
            - Hãy chắc chắn bạn đã cấp quyền truy cập webcam khi được hỏi
            """)


if __name__ == "__main__":
    import sys  # Import the sys module for accessing command-line arguments

    # Check if a model name is provided as a command-line argument
    args = len(sys.argv)
    model = sys.argv[1] if args > 1 else None  # Assign first argument as the model name if provided
    # Create an instance of the Inference class and run inference
    Inference(model=model).inference()