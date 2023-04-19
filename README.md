# Object-detection-using-YOLO-v8

## Data preparation:
1) Collect a dataset of images that contain the objects you want to detect.
2) Annotate the objects in the images using a tool like LabelImg or CVAT.
3) Convert the annotations to the YOLO format. This involves creating a text file for each image that lists the objects in the image along with their class label and bounding box coordinates.
## Training:
1) Install the YOLOv8 repository on your system.
2) Download a pre-trained YOLOv5 model from the official repository.
3) Modify the configuration file to match your dataset and training preferences.
4) Train the model using the following command: python train.py --img [image_size] --batch [batch_size] --epochs [num_epochs] --data [path_to_data.yaml] --cfg [path_to_config.yaml] --weights [path_to_pretrained_weights]
5) Monitor the training progress using the Tensorboard visualization tool.
## Evaluation:
1) Evaluate the trained model on a test dataset using the following command: python val.py --data [path_to_data.yaml] --weights [path_to_trained_weights] --task [task_type]
2) The evaluation will produce various metrics, such as mean average precision (mAP), which indicate how well the model performs.
## Inference:
1) Use the trained model to detect objects in new images or videos using the following command: python detect.py --weights [path_to_trained_weights] --img [path_to_image] --conf [confidence_threshold]
2) The output will be an image with bounding boxes around the detected objects and their corresponding class labels.
