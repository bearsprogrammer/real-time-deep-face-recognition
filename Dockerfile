FROM python:3.5-slim
# Installing curl to download the pre-trained model, det1.npy, det2.npy, det3.npy and requirements.txt - Installing g++, libglib2.0-0, libsm6, libxrender1, libxext6 which are required for the installment of grpcio, which is required for the installment of python packages by pip - Installing libgl1 which is required by cv2 package.
RUN apt-get update && apt-get install -y curl g++ libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 && rm -rf /var/cache/apt/archives /var/lib/apt/lists/*
# Downloading pre-trained model, required .npy files and requirements.txt.
RUN curl -L -O https://github.com/barisgecer/facegan/raw/master/facenet_model/20170511-185253.pb && curl -L -O https://github.com/davidsandberg/facenet/raw/master/src/align/det[1-3].npy && curl -L -O https://github.com/davidsandberg/facenet/raw/master/requirements.txt
RUN python3 -m pip install -r requirements.txt
# misc.imresize is deprecated in scipy package with versions newer than 1.1.0.
RUN pip uninstall -y scipy && pip install scipy==1.1.0
COPY . .
RUN python3 Make_aligndata_git.py && python3 Make_classifier_git.py
ENTRYPOINT ["python3"]
CMD ["realtime_facenet_git.py"]
