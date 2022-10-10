from boardcomms.coral_coms.coms_py.coral_pb_out import send_dx_dy

from inference.detector import Detector

import gstreamer



def main():
    model_path = "../model/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite"
    labels_path = "../model/coco_labels.txt"
    input_image_path = "../images/cats2.jpg"
    output_image_path = "../images/Doutput.jpg"


    camlist = pygame.camera.list_cameras()

    d1 = Detector(model_path, labels_path, .4)
    print(object_list)
    
    def user_callback(input_tensor, src_size, inference_box):
        start_time = time.monotonic()
        # run_inference(interpreter, input_tensor)
        # For larger input image sizes, use the edgetpu.classification.engine for better performance
        object_list = d1.run_detection(input_tensor, output_image_path)

        objs = get_objects(interpreter, args.threshold)[:args.top_k]
        end_time = time.monotonic()

        print('Inference: {:.2f} ms'.format((end_time - start_time) * 1000))
        return generate_svg(src_size, inference_box, objs, labels, text_lines)

    result = gstreamer.run_pipeline(user_callback,
                                    src_size=(640, 480),
                                    appsink_size=inference_size,
                                    videosrc=args.videosrc,
                                    videofmt=args.videofmt)

if __name__ == "__main__":
    main()