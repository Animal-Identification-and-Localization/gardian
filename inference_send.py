from boardcomms.coral_coms.coms_py.coral_pb_out import send_dx_dy

from inference.detector import Detector
import pygame
import pygame.camera
from pygame.locals import *

def main():
    model_path = "../model/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite"
    labels_path = "../model/coco_labels.txt"
    input_image_path = "../images/cats2.jpg"
    output_image_path = "../images/Doutput.jpg"

    pygame.init()
    pygame.camera.init()
    cam = pygame.camera.Camera("/dev/video1",(640,480))

    camlist = pygame.camera.list_cameras()

    d1 = Detector(model_path, labels_path, .4)
    object_list = d1.run_detection(input_image_path, output_image_path)
    print(object_list)
    

if __name__ == "__main__":
    main()