import os
import io
import qrcode
import uuid
import sys
import yaml
import traceback
import cv2
from flask import Flask, render_template, make_response, flash, send_file, request
from PIL import Image
import numpy as np

# Add HEIC support
from pillow_heif import register_heif_opener
register_heif_opener()

with open('./config.yaml', 'r') as fd:
    opts = yaml.safe_load(fd)

sys.path.insert(0, './white_box_cartoonizer/')

from cartoonize import WB_Cartoonize

if not opts['run_local']:
    if 'GOOGLE_APPLICATION_CREDENTIALS' in os.environ:
        from gcloud_utils import upload_blob, generate_signed_url, delete_blob
    else:
        raise Exception("GOOGLE_APPLICATION_CREDENTIALS not set in environment variables")

app = Flask(__name__)
app.secret_key = 'I m BaTmAn<>'
app.config['UPLOAD_FOLDER_VIDEOS'] = 'static/uploaded_videos'
app.config['UPLOAD_FOLDER_IMAGES'] = 'static/uploaded_images'
app.config['CARTOONIZED_FOLDER'] = 'static/cartoonized_outputs'
app.config['ALLOWED_IMAGE_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'heic'}
app.config['ALLOWED_VIDEO_EXTENSIONS'] = {'mp4', 'avi', 'mov'}
app.config['OPTS'] = opts

os.makedirs(app.config['UPLOAD_FOLDER_VIDEOS'], exist_ok=True)
os.makedirs(app.config['UPLOAD_FOLDER_IMAGES'], exist_ok=True)
os.makedirs(app.config['CARTOONIZED_FOLDER'], exist_ok=True)

wb_cartoonizer = WB_Cartoonize(os.path.abspath("white_box_cartoonizer/saved_models/"), opts['gpu'])

def convert_bytes_to_image(img_bytes):
    pil_image = Image.open(io.BytesIO(img_bytes))
    if pil_image.mode == "RGBA":
        image = Image.new("RGB", pil_image.size, (255, 255, 255))
        image.paste(pil_image, mask=pil_image.split()[3])
    else:
        image = pil_image.convert('RGB')
    image = np.array(image)
    return image

def allowed_file(filename, file_type):
    if file_type == 'image':
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_IMAGE_EXTENSIONS']
    elif file_type == 'video':
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_VIDEO_EXTENSIONS']
    return False

@app.route('/')
@app.route('/cartoonize', methods=["POST", "GET"])
def cartoonize():
    opts = app.config['OPTS']
    if request.method == 'POST':
        try:
            if request.files.get('image'):
                file = request.files["image"]
                if file.filename == '':
                    flash("No file selected", "error")
                    return render_template("index_cartoonized.html")

                filename = str(uuid.uuid4()) + "_" + file.filename
                file_ext = filename.rsplit('.', 1)[1].lower()

                if allowed_file(filename, 'image'):
                    upload_path = os.path.join(app.config['UPLOAD_FOLDER_IMAGES'], filename)
                    file.save(upload_path)

                    if file_ext == 'gif':
                        cartoonized_file_name = os.path.join(app.config['CARTOONIZED_FOLDER'], filename)
                        wb_cartoonizer.cartoonize_gif(upload_path, cartoonized_file_name)

                        if not opts["run_local"]:
                            output_uri = upload_blob("cartoonized_images", cartoonized_file_name, filename, content_type='image/gif')
                            os.system("rm " + cartoonized_file_name)
                            cartoonized_file_name = generate_signed_url(output_uri)
                    else:
                        with open(upload_path, 'rb') as f:
                            img = f.read()
                        image = convert_bytes_to_image(img)

                        cartoon_image = wb_cartoonizer.infer(image)
                        cartoonized_file_name = os.path.join(app.config['CARTOONIZED_FOLDER'], filename + ".jpg")
                        cv2.imwrite(cartoonized_file_name, cv2.cvtColor(cartoon_image, cv2.COLOR_RGB2BGR))

                        if not opts["run_local"]:
                            output_uri = upload_blob("cartoonized_images", cartoonized_file_name, filename + ".jpg", content_type='image/jpg')
                            os.system("rm " + cartoonized_file_name)
                            cartoonized_file_name = generate_signed_url(output_uri)

                    os.remove(upload_path)
                    return render_template("index_cartoonized.html", cartoonized_image=cartoonized_file_name)

                elif allowed_file(filename, 'video'):
                    upload_path = os.path.join(app.config['UPLOAD_FOLDER_VIDEOS'], filename)
                    file.save(upload_path)

                    cartoonized_video_path = os.path.join(app.config['CARTOONIZED_FOLDER'], f"cartoonized_{filename}")
                    cartoonized_video_path = wb_cartoonizer.cartoonize_video(upload_path, cartoonized_video_path)

                    if not opts["run_local"]:
                        output_uri = upload_blob("cartoonized_videos", cartoonized_video_path, f"cartoonized_{filename}", content_type='video/mp4')
                        os.system("rm " + cartoonized_video_path)
                        cartoonized_video_path = generate_signed_url(output_uri)
                    else:
                        cartoonized_video_path = os.path.join(app.config['CARTOONIZED_FOLDER'], f"cartoonized_{filename}")

                    os.remove(upload_path)
                    return render_template("index_cartoonized.html", cartoonized_image=cartoonized_video_path)

                else:
                    flash("File type not allowed. Please upload an image (png, jpg, jpeg, gif, heic) or video (mp4, avi, mov).", "error")
                    return render_template("index_cartoonized.html")

        except Exception as e:
            print(traceback.print_exc())
            flash(f"Our server hiccuped :/ Please upload another file! :) Error: {str(e)}", "error")
            return render_template("index_cartoonized.html")
    else:
        return render_template("index_cartoonized.html")

@app.route('/cartoonized/<filename>')
def serve_cartoonized(filename):
    file_path = os.path.join(app.config['CARTOONIZED_FOLDER'], filename)
    if os.path.exists(file_path):
        if filename.endswith('.mp4'):
            return send_file(file_path, mimetype='video/mp4')
        elif filename.endswith(('.jpg', '.jpeg')):
            return send_file(file_path, mimetype='image/jpeg')
        elif filename.endswith('.png'):
            return send_file(file_path, mimetype='image/png')
        elif filename.endswith('.gif'):
            return send_file(file_path, mimetype='image/gif')
    return "File not found", 404

if __name__ == "__main__":
    try:
        import socket
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        url = f"http://{local_ip}:8080"

        print(f"\nYour Flask app is running at: {url}\n")

        qr = qrcode.make(url)
        qr.show()

    except Exception as e:
        print("Failed to generate QR code:", e)

    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
