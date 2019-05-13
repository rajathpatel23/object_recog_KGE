Name: Rajat Patel & Mohit Khatwani
Project: Combining Image Recognition with Knowledge Graph Embedding for Learning Semantic Attribute of Images
email ID: rpatel12@umbc.edu & khatwan1@umbc.edu

baseline_model_1 : The baseline model architecture implemented, the file is self sufficient to run, only changes required are file location there mentioned as global parameter
baseline_model_2: Proposed solution model architecture implemented, the file is self sufficient to run, only changes required are file location there mentioned as global parameter
VGG_encoder: The program gives encoded representation of the images and saves them as a numpy array
reading_file_loc: This file does the data preprocessing required for the model architectures and saves the required dataframe.


Running the object recognition model
yolo_recog.py: Main file to run the pretrained model, for object detection.
yolov3.py: Definition of YOLO model
yolov3.weights: weights of trained model on COCO dataset
convert_weights.py & convert_weights_pb.py: Convert weights file to protobuffer file


pickle_files:
label_to_imagedict.pickle : Dictionary with key: labels, values: list of image information from pycoco api
filename_imagedict.pickle : Dictionary with key: labels, values: file_names
label_to_object_final.pickle: Dictionary with key: labels, values: list of object detected in an image


encoded_image_vec.npz: encoded image vectors

train_vec_file: dataframe of word embedding and class labels


