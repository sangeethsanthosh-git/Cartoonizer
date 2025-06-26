# Cartoon
* This project refers to cartoonize a normal image or video
* In this video can be upload upto 30sec in any size upto 100mb .there is a gpu version for doing the process faster but every device didnot contain a gpu so this project is fully capable of cpu acceleration so no bother about gpu.
* Also image can be cartoonize within sec but video convertion an take upto 1.5 to 3 min because of frame per frame cartoonzing 

### Application tested on:

- python 3.9


### Using `venv`

1. Make a virtual environment using `venv` and activate it
```
venv\Scripts\activate

make sure the file location is correct the project is with in a folder (eg:cd C:\Project2025\cartoonize\cartoonize) 
```
2. Install python dependencies
```
pip install -r requirements.txt
```
3. Run the webapp. Be sure to set the appropriate values in `config.yaml` file before running the application.
```
python app.py
```
## Installation

### you can use pip install -r requirements.txt for installing package or else this can use

pip install absl-py algorithmia algorithmia-api-client astor astunparse cached-property cachetools certifi charset-normalizer clang click colorama Flask flask-ngrok flatbuffers gast google-api-core google-auth google-auth-oauthlib google-cloud-core google-cloud-storage google-pasta google-resumable-media googleapis-common-protos grpcio gunicorn h5py idna importlib-metadata itsdangerous Jinja2 keras Keras-Applications Keras-Preprocessing libclang Markdown MarkupSafe numpy oauthlib opencv-python opt-einsum packaging Pillow pip proto-plus protobuf pyasn1 pyasn1-modules python-dateutil PyYAML requests requests-oauthlib rsa scikit-video scipy setuptools six sk-video tensorboard tensorboard-data-server tensorboard-plugin-wit tensorflow tensorflow-cpu tensorflow-estimator tensorflow-intel tensorflow-io-gcs-filesystem termcolor tf-slim typing-extensions urllib3 Werkzeug wheel wrapt zipp

### maybe after installation the project couldn't work it shows tensorflow error because of version inconsistancy so use following installing also 

1. pip uninstall -y tensorflow tensorflow-cpu tensorflow-intel protobuf

2. pip list | findstr "tensorflow protobuf ml-dtypes tensorboard"

3. pip uninstall -y ml-dtypes tensorboard tensorboard-data-server tensorboard-plugin-wit tensorflow-estimator tensorflow-io-gcs-filesystem   

4. pip install tensorflow==2.10.0 protobuf==3.20.3 ml-dtypes==0.4.0 tensorboard==2.18.0    

5. pip uninstall -y tensorflow tensorboard protobuf 

6. pip uninstall -y tensorflow-estimator tensorboard protobuf    

7. pip uninstall flask opencv-python pillow numpy tensorflow googleapis-common-protos -y

8. pip install flask opencv-python pillow numpy==1.24.3 protobuf==3.20.3 googleapis-common-protos==1.69.2    

9. pip install tensorflow-cpu==2.10.0 protobuf==3.19.6 tensorboard==2.10.1

10. pip uninstall protobuf    

11. pip install protobuf==3.20.3   

12. pip uninstall protobuf googleapis-common-protos -y   

13. pip install protobuf==3.19.6 googleapis-common-protos==1.56.4

### after installing this may be error show in the problem section in vs code no problem because of catch of previous work don't worry after installing all just excute the program 

python app.py

### remember that the virtual environment (venv) is active  

okay that's all happy coding  ^_^