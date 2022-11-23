
import argparse
import camera_pipeline
import os
import time
import atexit

from common import avg_fps_counter, SVG
from pycoral.adapters.common import input_size
from pycoral.adapters.detect import get_objects
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.edgetpu import run_inference
from boardcomms.coral_coms.coms_py.coral_pb_out import send_dx_dy
from periphery import I2C

def generate_svg(src_size, inference_box, objs, labels, text_lines):
    svg = SVG(src_size)
    src_w, src_h = src_size
    box_x, box_y, box_w, box_h = inference_box
    scale_x, scale_y = src_w / box_w, src_h / box_h

    for y, line in enumerate(text_lines, start=1):
        svg.add_text(10, y * 20, line, 20)
    for obj in objs:
        bbox = obj.bbox
        if not bbox.valid:
            continue
        # Absolute coordinates, input tensor space.
        x, y = bbox.xmin, bbox.ymin
        w, h = bbox.width, bbox.height
        # Subtract boxing offset.
        x, y = x - box_x, y - box_y
        # Scale to source coordinate space.
        x, y, w, h = x * scale_x, y * scale_y, w * scale_x, h * scale_y
        percent = int(100 * obj.score)
        label = '{}% {}'.format(percent, labels.get(obj.id, obj.id))
        svg.add_text(x, y - 5, label, 20)
        svg.add_rect(x, y, w, h, 'red', 2)
    return svg.finish()

def get_bin(dx, dy, inference_size):
  print(f'{dx}, {dy}')
  bin_x = int(float(dx)/inference_size[0]*40)-20
  bin_y = int(float(dy)/inference_size[1]*40)-20
  return (bin_x, bin_y)

i2c = None
def main():
    default_model_dir = './models'
    default_model = 'output_tflite_graph_edgetpu-500.tflite'
    default_labels = 'labels_500.txt'
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='.tflite model path',
                        default=os.path.join(default_model_dir,default_model))
    parser.add_argument('--labels', help='label file path',
                        default=os.path.join(default_model_dir, default_labels))
    parser.add_argument('--top_k', type=int, default=3,
                        help='number of categories with highest score to display')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='classifier score threshold')
    parser.add_argument('--videosrc', help='Which video source to use. ',
                        default='/dev/video0')
    parser.add_argument('--no_spi', help='do not send coordinates over SPI',
                        action='store_true')
    parser.add_argument('--display', help='open display window for inference',
                        action='store_true')
    args = parser.parse_args()

    print('Loading {} with {} labels.'.format(args.model, args.labels))
    interpreter = make_interpreter(args.model)
    interpreter.allocate_tensors()
    labels = read_label_file(args.labels)
    inference_size = input_size(interpreter)
    print(inference_size)

    # Average fps over last 30 frames.
    fps_counter = avg_fps_counter(30)
    no_spi = args.no_spi

    i2c = I2C("/dev/i2c-3")
    prev_dx = -1000
    prev_dy = -1000

    def user_callback(input_tensor, src_size, inference_box):
      nonlocal fps_counter
      nonlocal no_spi
      nonlocal prev_dx
      nonlocal prev_dy
      
      start_time = time.monotonic()
      run_inference(interpreter, input_tensor)
      end_time = time.monotonic()
      
      # For larger input image sizes, use the edgetpu.classification.engine for better performance
      objs = get_objects(interpreter, args.threshold)[:args.top_k]

      print(objs)
      if(len(objs)>0):
        print(objs[0].id)
        dx = int((objs[0].bbox.xmax+objs[0].bbox.xmin)/2)
        dy = int((objs[0].bbox.ymax+objs[0].bbox.ymin)/2)
        dx, dy = get_bin(dx, dy, inference_size)
        print(f'dx: {dx}, dy: {dy}')
        if not no_spi and (prev_dx != dx or prev_dy != dy): 
          try: 
            print('sending coordinates to arduino')
            send_dx_dy(dx*5, dy*5, i2c)
          except:
            print('i2c target busy')
            time.sleep(.25)
          prev_dx = dx
          prev_dy = dy

        print(inference_box)
        print(src_size)
      text_lines = [
          'Inference: {:.2f} ms'.format((end_time - start_time) * 1000),
          'FPS: {} fps'.format(round(next(fps_counter))),
      ]
      print(' '.join(text_lines))
      return generate_svg(src_size, inference_box, objs, labels, text_lines)
    headless = not args.display
    print(headless)
    result = camera_pipeline.run_pipeline(user_callback,
                                    src_size=(640, 480),
                                    appsink_size=inference_size,
                                    videosrc=args.videosrc,
                                    headless=headless)

def exit_handler():
  print('Closing i2c')
  i2c.close()

if __name__ == '__main__':
  atexit.register(exit_handler)
  main()
