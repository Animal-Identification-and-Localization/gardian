from boardcomms.coral_coms.coms_py.coral_pb_out import send_dx_dy

from inference.detector import Detector

import gstreamer



def main():
    model_path = "models/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite"
    labels_path = "labels/coco_labels.txt"
    input_image_path = "pics/coral_camera_0000.jpg"
    output_image_path = "pics/Doutput.jpg"

    d1 = Detector(model_path, labels_path, .4)
    object_list = d1.run_detection(input_image_path, output_image_path)
    
    if object_list is not None:
        for obj in object_list:
            print(obj)
            bounding_box = obj.bbox
            id = obj.id
            score = obj.score
            x0 = bounding_box.xmin
            x1 = bounding_box.xmax
            y0 = bounding_box.ymin
            y1 = bounding_box.ymax

            dx = int((x0+x1)/2)
            dy = int((y1+y0)/2)

            printf(f'Sending {id} at ({}, {}) to ATMega')
            send_dx_dy(dx, dy)

    

if __name__ == "__main__":
    main()