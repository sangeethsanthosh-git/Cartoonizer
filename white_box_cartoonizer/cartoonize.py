"""
Internal code snippets were obtained from https://github.com/SystemErrorWang/White-box-Cartoonization/

For it to work tensorflow version 2.x changes were obtained from https://github.com/steubk/White-box-Cartoonization 
"""
import os
import sys
import logging

import cv2
import numpy as np
from PIL import Image
try:
    import tensorflow.compat.v1 as tf
except ImportError:
    import tensorflow as tf

import network
import guided_filter

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WB_Cartoonize:
    def __init__(self, weights_dir, gpu):
        if not os.path.exists(weights_dir):
            raise FileNotFoundError("Weights Directory not found, check path")
        self.gpu = gpu
        self.load_model(weights_dir)
        print("Weights successfully loaded")
        # Log GPU availability
        print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
        print("Physical Devices: ", tf.config.list_physical_devices('GPU'))

    def resize_crop(self, image):
        h, w, c = np.shape(image)
        if min(h, w) > 720:
            if h > w:
                h, w = int(720*h/w), 720
            else:
                h, w = 720, int(720*w/h)
        image = cv2.resize(image, (w, h),
                            interpolation=cv2.INTER_AREA)
        h, w = (h//8)*8, (w//8)*8
        image = image[:h, :w, :]
        return image

    def load_model(self, weights_dir):
        try:
            tf.disable_eager_execution()
        except:
            None

        tf.reset_default_graph()

        self.input_photo = tf.placeholder(tf.float32, [1, None, None, 3], name='input_image')
        network_out = network.unet_generator(self.input_photo)
        self.final_out = guided_filter.guided_filter(self.input_photo, network_out, r=1, eps=5e-3)

        all_vars = tf.trainable_variables()
        gene_vars = [var for var in all_vars if 'generator' in var.name]
        saver = tf.train.Saver(var_list=gene_vars)
        
        if self.gpu:
            gpu_options = tf.GPUOptions(allow_growth=True)
            device_count = {'GPU': 1}
        else:
            gpu_options = None
            device_count = {'GPU': 0}
        
        config = tf.ConfigProto(gpu_options=gpu_options, device_count=device_count)
        
        self.sess = tf.Session(config=config)

        self.sess.run(tf.global_variables_initializer())
        saver.restore(self.sess, tf.train.latest_checkpoint(weights_dir))

    def infer(self, image):
        image = self.resize_crop(image)
        batch_image = image.astype(np.float32)/127.5 - 1
        batch_image = np.expand_dims(batch_image, axis=0)
        
        # Session Run
        output = self.sess.run(self.final_out, feed_dict={self.input_photo: batch_image})
        
        # Post Process
        output = (np.squeeze(output)+1)*127.5
        output = np.clip(output, 0, 255).astype(np.uint8)
        
        return output

    def cartoonize_gif(self, input_gif_path, output_gif_path):
        """Convert a GIF to a cartoon-style GIF by processing every frame."""
        gif = Image.open(input_gif_path)
        frames = []
        durations = []
        frame_idx = 0
        speed_up_factor = 0.25  # Speed up 4x to match original speed

        try:
            while True:
                # Convert frame to RGB
                frame = gif.convert('RGB')
                frame_np = np.array(frame)
                
                # Process every frame for maximum smoothness
                cartoon_frame = self.infer(frame_np)

                # Convert back to PIL Image
                cartoon_frame_pil = Image.fromarray(cartoon_frame)
                frames.append(cartoon_frame_pil)
                # Adjust duration to speed up playback
                original_duration = gif.info.get('duration', 100)
                adjusted_duration = int(original_duration * speed_up_factor)
                durations.append(max(adjusted_duration, 10))  # Ensure minimum duration for compatibility
                
                frame_idx += 1
                gif.seek(gif.tell() + 1)
        except EOFError:
            pass

        # Save the cartoonized GIF
        frames[0].save(
            output_gif_path,
            save_all=True,
            append_images=frames[1:],
            loop=0,  # Loop indefinitely
            duration=durations  # Use adjusted durations
        )

        print(f"Cartoonized GIF saved as '{output_gif_path}'")
        return output_gif_path

    def cartoonize_video(self, input_video_path, output_video_path):
        """Convert a video to a cartoon-style video by processing every 2nd frame."""
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file '{input_video_path}'.")

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Downscale resolution to 1280x720 if larger
        max_width = 1280  # Target maximum width
        if width > max_width:
            scale = max_width / width
            width = int(width * scale)
            height = int(height * scale)
            width = (width // 2) * 2  # Ensure even dimensions
            height = (height // 2) * 2

        print(f"Video Info: FPS={fps}, Width={width}, Height={height}, Total Frames={total_frames}")

        # Use H.264 codec for better browser compatibility with fallback
        fourcc = cv2.VideoWriter_fourcc(*'avc1')  # Primary H.264 codec
        output_video_path = output_video_path.rsplit('.', 1)[0] + '.mp4'  # Use .mp4 extension
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        if not out.isOpened():
            logger.warning("Failed to open VideoWriter with avc1 codec, trying mp4v fallback")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Fallback H.264 codec
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
            if not out.isOpened():
                cap.release()
                raise ValueError(f"Could not create output video file '{output_video_path}' with either avc1 or mp4v codec. Check OpenCV FFmpeg support.")

        frame_idx = 0
        last_cartoon_frame = None
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Resize frame to match output resolution
            if width != int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)):
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

            frame_idx += 1
            print(f"Processing frame {frame_idx}/{total_frames}...")

            # Process every 2nd frame for better smoothness
            if frame_idx % 2 == 1:  # Process every 2nd frame
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                cartoon_frame = self.infer(frame_rgb)
                last_cartoon_frame = cartoon_frame
            else:  # Use the last cartoonized frame for other frames
                cartoon_frame = last_cartoon_frame if last_cartoon_frame is not None else frame

            cartoon_frame_bgr = cv2.cvtColor(cartoon_frame, cv2.COLOR_RGB2BGR)
            out.write(cartoon_frame_bgr)

        cap.release()
        out.release()
        cv2.destroyAllWindows()  # Ensure all OpenCV resources are released

        # Verify the video file is created and playable with detailed logging
        if os.path.exists(output_video_path) and os.path.getsize(output_video_path) > 0:
            logger.info(f"Cartoonized video saved successfully as '{output_video_path}' with size {os.path.getsize(output_video_path)} bytes")
        else:
            logger.error(f"Failed to create valid video file at '{output_video_path}' with size {os.path.getsize(output_video_path) if os.path.exists(output_video_path) else 0} bytes")
            raise ValueError(f"Failed to create valid video file at '{output_video_path}'")
        return output_video_path

if __name__ == '__main__':
    gpu = len(sys.argv) < 2 or sys.argv[1] != '--cpu'
    wbc = WB_Cartoonize(os.path.abspath('white_box_cartoonizer/saved_models'), gpu)
    img = cv2.imread('white_box_cartoonizer/test.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cartoon_image = wbc.infer(img)
    import matplotlib.pyplot as plt
    plt.imshow(cartoon_image)
    plt.show()