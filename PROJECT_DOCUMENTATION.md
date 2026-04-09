# Project Documentation: Cartoonizer

## 1. Project Overview

Cartoonizer is a web-based media stylization project built with Flask, TensorFlow, OpenCV, and Pillow. Its purpose is to convert ordinary visual media into a cartoon-style output using a pretrained White-box Cartoonization model.

The application allows a user to upload:

- static images
- animated GIFs
- short videos

After upload, the system processes the file, applies the cartoonization pipeline, and returns a downloadable result through a browser-based interface.

## 2. Problem Statement

Creating cartoon-style visual content manually takes time and usually requires editing tools or artistic skill. This project automates that process by applying a neural image transformation pipeline that produces stylized outputs from user-provided media.

The main goal is to make cartoonization accessible through a simple upload-and-download workflow.

## 3. Objectives

- Build an easy-to-use web application for cartoonizing media.
- Support multiple file types in one interface.
- Provide local execution without requiring cloud infrastructure.
- Keep the project compatible with CPU-only systems, while allowing GPU acceleration when available.
- Reuse a pretrained deep learning model rather than training a model from scratch.

## 4. Core Features

- Upload support for images, GIFs, and videos.
- Cartoonization of `png`, `jpg`, `jpeg`, `gif`, and `heic` images.
- Cartoonization of `mp4`, `avi`, and `mov` videos.
- Automatic saving of generated outputs into the `static/` directory.
- Result preview and download from the browser.
- Local network URL generation with QR code output at app startup.
- Optional GPU usage through configuration.

## 5. Technology Stack

- Backend: Flask
- Deep Learning Framework: TensorFlow 2.10 with `tf.compat.v1`
- Image and Video Processing: OpenCV
- Image Handling: Pillow and `pillow-heif`
- Numerical Processing: NumPy
- Frontend: HTML, Tailwind CSS, JavaScript, Jinja templates
- Runtime Target: Python 3.9

## 6. High-Level Architecture

The project is organized into three main layers:

### 6.1 Presentation Layer

The user interface is defined in `templates/index_cartoonized.html`. It provides:

- a file upload button
- a loader while processing is in progress
- preview support for image and video outputs
- a download action for the generated file

### 6.2 Application Layer

The main web application lives in `app.py`. This layer is responsible for:

- accepting file uploads through Flask routes
- validating allowed file extensions
- saving uploaded files temporarily
- selecting the correct processing path for image, GIF, or video input
- returning the generated output to the user

### 6.3 Model and Processing Layer

The deep learning pipeline is implemented inside `white_box_cartoonizer/`.

- `cartoonize.py` wraps the model and exposes inference methods
- `network.py` defines the neural network architecture
- `guided_filter.py` applies guided filtering to refine the output
- `saved_models/` contains pretrained model weights

## 7. Functional Workflow

### 7.1 Image Processing

1. The user uploads an image through the web page.
2. Flask stores the uploaded file in `static/uploaded_images/`.
3. The file is read into memory and converted into an RGB NumPy array.
4. The White-box Cartoonization model performs inference.
5. The result is saved into `static/cartoonized_outputs/`.
6. The generated image is displayed back to the user.

### 7.2 GIF Processing

1. The uploaded GIF is opened frame by frame.
2. Each frame is converted to RGB.
3. Each frame is passed through the inference pipeline.
4. The processed frames are combined into a new GIF.
5. The new GIF is saved and returned for preview/download.

### 7.3 Video Processing

1. The uploaded video is stored in `static/uploaded_videos/`.
2. OpenCV reads the file and extracts video properties.
3. Frames are resized if the resolution is too large.
4. The system cartoonizes alternating frames and reuses the last processed frame for the rest to reduce computation cost.
5. The output is written as an MP4 when possible.
6. The generated video is displayed back to the user.

## 8. Project Structure

```text
Cartoonizer/
|-- app.py
|-- config.yaml
|-- requirements.txt
|-- runtime.txt
|-- video_api.py
|-- README.md
|-- PROJECT_DOCUMENTATION.md
|-- assets/
|   |-- input.png
|   `-- output.png
|-- templates/
|   `-- index_cartoonized.html
|-- static/
|   |-- uploaded_images/
|   |-- uploaded_videos/
|   |-- cartoonized_outputs/
|   `-- sample_images/
`-- white_box_cartoonizer/
    |-- cartoonize.py
    |-- network.py
    |-- guided_filter.py
    `-- saved_models/
```

## 9. Configuration

The runtime behavior is controlled through `config.yaml`.

Important keys include:

- `run_local`
- `gpu`
- `trim-video`
- `trim-video-length`
- `original_frame_rate`
- `output_frame_rate`
- `original_resolution`
- `resize-dim`

In the current codebase, `run_local` and `gpu` are the most important active settings for the main Flask flow.

## 10. Installation and Execution

### 10.1 Recommended Environment

- Python 3.9
- Virtual environment enabled

### 10.2 Setup Steps

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install pillow-heif
python app.py
```

After startup, the app runs on port `8080` by default.

## 11. Supported File Types

### Image Formats

- `.png`
- `.jpg`
- `.jpeg`
- `.gif`
- `.heic`

### Video Formats

- `.mp4`
- `.avi`
- `.mov`

## 12. Output Behavior

- Static images are usually exported as `.jpg`
- GIFs are exported as `.gif`
- Videos are written as `.mp4` when codec support is available
- Generated results are stored inside `static/cartoonized_outputs/`

## 13. Strengths of the Project

- Supports multiple media types in one application.
- Combines deep learning inference with a simple web interface.
- Works locally without mandatory cloud deployment.
- Includes practical enhancements such as HEIC support and QR-based local access.
- Uses pretrained weights, which makes the app usable without a model training phase.

## 14. Current Limitations

- Video processing is slower than image processing because it is frame-based.
- TensorFlow and dependency compatibility are best with Python 3.9.
- Some cloud-related paths in `app.py` depend on helper modules not present in this repository snapshot.
- Some entries in `config.yaml` appear to belong to older or partially integrated video-processing logic.
- Browser compatibility for output video depends on available codecs in the local OpenCV/FFmpeg build.

## 15. Possible Future Improvements

- Add drag-and-drop uploads and richer progress reporting.
- Add batch processing for multiple images.
- Add background job handling for long video tasks.
- Improve deployment support for cloud hosting.
- Add automated tests for upload validation and route behavior.
- Add cleanup jobs for temporary files.
- Add model selection or style-strength controls.

## 16. Conclusion

Cartoonizer is a practical deep learning application that demonstrates how neural style transfer concepts can be turned into a user-facing product. It combines model inference, media handling, and web delivery in a compact Flask project. The result is a useful portfolio-ready application that highlights computer vision integration, media processing, and end-to-end product thinking.
