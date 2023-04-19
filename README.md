# Object-detection-using-YOLO-v8

## Data preparation:
Collect a dataset of images that contain the objects you want to detect.
Annotate the objects in the images using a tool like LabelImg or CVAT.
Convert the annotations to the YOLO format. This involves creating a text file for each image that lists the objects in the image along with their class label and bounding box coordinates.
## Training:
Install the YOLOv5 repository on your system.
Download a pre-trained YOLOv5 model from the official repository.
Modify the configuration file to match your dataset and training preferences.
Train the model using the following command: python train.py --img [image_size] --batch [batch_size] --epochs [num_epochs] --data [path_to_data.yaml] --cfg [path_to_config.yaml] --weights [path_to_pretrained_weights]
Monitor the training progress using the Tensorboard visualization tool.
## Evaluation:
Evaluate the trained model on a test dataset using the following command: python val.py --data [path_to_data.yaml] --weights [path_to_trained_weights] --task [task_type]
The evaluation will produce various metrics, such as mean average precision (mAP), which indicate how well the model performs.
## Inference:
Use the trained model to detect objects in new images or videos using the following command: python detect.py --weights [path_to_trained_weights] --img [path_to_image] --conf [confidence_threshold]
The output will be an image with bounding boxes around the detected objects and their corresponding class labels.
