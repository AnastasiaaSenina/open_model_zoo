# Human Pose Estimation Demo (top-down pipeline)

This demo showcases top-down pipeline for human pose estimation on video or image. The task is to predict bboxes for everyone person on frame and then to predict a pose for everyone detected person. The pose may contain up to 17 keypoints: ears, eyes, nose, shoulders, elbows, wrists, hips, knees, and ankles.

# How It Works

On the start-up, the application reads command line parameters and loads  detection person model and single human pose estimation model. Upon getting a frame from the OpenCV VideoCapture, then  demo executes top/down pipeline for this frame and displays the results.

# Running

Running the application with the `-h` option yields the following usage message:
```
usage: demo.py [-h] -m_od MODEL_OD_XML -m_hpe MODEL_HPE_XML
               [-i INPUT [INPUT ...]] [-d DEVICE]
               [--cpu-extension CPU_EXTENSION] [--label-person LABEL_PERSON]
               [--no_show]

optional arguments:
  -h, --help            show this help message and exit
  -m_od MODEL_OD_XML, --model-od-xml MODEL_OD_XML
                        path to model of object detector in xml format
  -m_hpe MODEL_HPE_XML, --model-hpe-xml MODEL_HPE_XML
                        path to model of human pose estimator in xml format
  -i INPUT [INPUT ...], --input INPUT [INPUT ...]
                        path to video or image/images
  -d DEVICE, --device DEVICE
                        Specify the target to infer on CPU or GPU
  --cpu-extension CPU_EXTENSION
                        path to cpu extension
  --label-person LABEL_PERSON
                        Label of class person for detector
  --no_show             Optional. Do not display output.

```
To run the demo, you can use public or pre-trained models. To download the pre-trained models, use the OpenVINO Model Downloader or go to [https://download.01.org/opencv/](https://download.01.org/opencv/).

For example, to do inference on a CPU, run the following command:

```sh
 python demo.py  --model-od-xml <path_to_dir__with_models>/MobileNetSSD_deploy.xml --model-od-bin <path_to_dir__with_models>/MobileNetSSD_deploy.bin --model-hpe-bin <path_to_dir__with_models>/single-human-pose-estimation-0001.bin --model-hpe-xml /home/inteladmin/single-human-pose-estimation-0001.xml --video <path_to_video>/back-passengers.avi --cpu_extension <path_to_lib>/libcpu_extension_avx2.so
```

The demo uses OpenCV to display the resulting frame with estimated poses and text report in format summary FPS / FPS single human pose/ FPS detector.