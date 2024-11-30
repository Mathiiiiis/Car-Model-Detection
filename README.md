# Car Model Detection


# 🚗 Project Overview
This project aims to build an application capable of detecting vehicles in images and identifying their exact brand and model. Using object detection and classification algorithms, the system provides accurate results, making it suitable for car enthusiasts, traffic monitoring, or similar professional applications.


# 🛠 Features
Object Detection: Localizes vehicles in the input images using YOLO.

Vehicle Recognition: Identifies the exact brand/model of the detected vehicles using a ResNet-based neural network.


# 📊 Results and Performance
Classification Accuracy: ~70% on a custom dataset of 19 car brands.

Object Detection: YOLO models (YOLOv5s and YOLOv8n) are used for high-speed and accurate vehicle detection.

Supported car brands:

Audi, Bentley, Benz, BMW, Cadillac, Dodge, Ferrari, Ford, Ford Mustang, Kia, Lamborghini, Lexus, Maserati, Porsche, Rolls Royce, Tesla, Toyota, Alfa Romeo, Hyundai.



# 📚 File Structure 

![image](https://github.com/user-attachments/assets/a19f9d5e-33e1-416a-82e4-1c9ed9642795)

# 🔍 Example Output
Before : 

![image](https://github.com/user-attachments/assets/749f221d-a704-4151-9a37-25f62ec3d3b6)

After :

![image](https://github.com/user-attachments/assets/1243c841-9191-4bed-93af-a89f8b6a6716)


# 🚀 Usage
Predict the car model in an image

Update the image_path in main.py with the path to your image.

Run the code : main.py 

⚠️ Take care to change the link on line 74. Replace the image destination with yours⚠️ 

The output will display the car model and draw a bounding box around the detected vehicle


# 📚 References
PyTorch

YOLO : https://github.com/ultralytics/yolov5

TorchVision Models


# 🔧 Future Work

Improve classification accuracy by augmenting the dataset.

Add support for more car brands and models.
Implement real-time detection via webcam or video input.
