
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
from gi.repository import Gst
from periphery import I2C

def generate_svg(src_size, inference_box, objs, labels, text_lines):
    svg = SVG(src_size)
    src_w, src_h = src_size
    box_x, box_y, box_w, box_h = inference_box
    scale_x, scale_y = src_w / box_w, src_h / box_h


    for obj in objs:
        bbox = obj.bbox

        # person detected
        if obj.id == 1:
          text_lines.append('Human detected, sleeping...')
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
        if obj.id == 2: svg.add_rect(x, y, w, h, 'red', 2)
        else: svg.add_rect(x, y, w, h, 'green', 2)

    for y, line in enumerate(text_lines, start=1):
        svg.add_text(10, y * 20, line, 20)
    return svg.finish()

def get_bin(dx, dy, inference_size):
  bin_x = int(float(dx)/inference_size[0]*7)-3
  bin_y = int(float(dy)/inference_size[1]*5)-2
  return (bin_x, bin_y)

def get_area(bbox):
  x = int(bbox.xmax-bbox.xmin)
  y = int(bbox.ymax-bbox.ymin)
  return x*y

  
i2c = None
def main():
    global i2c
    default_model_dir = './models'
    default_model = 'output_tflite_graph_edgetpu_15.tflite'
    default_labels = 'labels.txt'
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
    parser.add_argument('--laser', help='Turn on laser pointer',
                    action='store_true')
    parser.add_argument('--display', help='open display window for inference',
                        action='store_true')
    args = parser.parse_args()

    print('Loading {} with {} labels.'.format(args.model, args.labels))
    interpreter = make_interpreter(args.model)
    interpreter.allocate_tensors()
    labels = read_label_file(args.labels)
    inference_size = input_size(interpreter)
    use_laser = args.laser
    print(inference_size)

    # Average fps over last 30 frames.
    fps_counter = avg_fps_counter(30)
    no_spi = args.no_spi

    i2c = I2C("/dev/i2c-3")
    prev_dx = -1000
    prev_dy = -1000
    headless = not args.display
    no_obj_count = 0
    bbox_max_size = int(inference_size[0]*inference_size[1]*.4)

    def user_callback(input_tensor, src_size, inference_box):
      nonlocal no_obj_count
      nonlocal fps_counter
      nonlocal no_spi
      nonlocal prev_dx
      nonlocal prev_dy
      nonlocal headless
      nonlocal use_laser
      nonlocal bbox_max_size

      start_time = time.monotonic()
      run_inference(interpreter, input_tensor)
      end_time = time.monotonic()
      
      # For larger input image sizes, use the edgetpu.classification.engine for better performance
      objs = get_objects(interpreter, args.threshold)[:args.top_k]
      text_lines = ''
      target = None
      fps = round(next(fps_counter)) 

      text_lines = [
        'Inference: {:.2f} ms'.format((end_time - start_time) * 1000),
        'FPS: {} fps'.format(fps),
      ]

      area_target = 300*300

      for idx in range(0,len(objs)):
        area = get_area(objs[idx].bbox)
        # print(f'area {area}, thres {bbox_max_size}')
        if area>bbox_max_size:
          objs[idx] = None

        elif objs[idx].id == 2 and area<area_target:
          target = objs[idx]
          area_target = area

        # if a human is found, sleep for 5s
        elif objs[idx].id == 1 and objs[idx].score>.65:
          if not no_spi: send_dx_dy(0,0,i2c)
          svg = generate_svg(src_size, inference_box, [objs[idx]], labels, text_lines)
          return (svg, True)

      # print(objs)


      if target is not None:
        no_obj_count = 0

        dx = int((target.bbox.xmax+target.bbox.xmin)/2)
        dy = int((target.bbox.ymax+target.bbox.ymin)/2)

        dx = int(int((dx - inference_size[0]/2)/5)*4)*int(abs(dx)>=25)
        if dx<0: dx = max(dx, -200)
        if dx>0: dx = min(dx, 200)

        dy = int(int((dy - inference_size[1]/2))/5)*int(abs(dy)>=25)

        if not no_spi: 
          try: 
            # print('sending coordinates to arduino')
            thres = 10
            laser_on = prev_dx<thres and dx<thres and dy<thres and prev_dy<thres and use_laser
            # print(laser_on)
            # start_time = time.monotonic()
            send_dx_dy(-1*dx, -1*int(dy), i2c, laser_on)
            # send_dx_dy(0, 0, i2c, laser_on)
            # end_time = time.monotonic()
            # print(f'{(end_time-start_time)*1000}')
          except:
            print('i2c target busy')

          
          prev_dx = dx
          prev_dy = dy

        # print(inference_box)
        # print(src_size)
        
      else:
          no_obj_count += 1

          if no_obj_count > 6:
              if not no_spi: send_dx_dy(0,0,i2c)
      

      # quit()

      objs = [i for i in objs if i is not None]
      return (generate_svg(src_size, inference_box, objs, labels, text_lines), False)
    
    headless = not args.display
    print(headless)
    print(inference_size)
    result = camera_pipeline.run_pipeline(user_callback,
                                    src_size=(640, 480),
                                    appsink_size=inference_size,
                                    videosrc=args.videosrc,
                                    headless=headless)

def exit_handler():
  print('Closing i2c')
  if i2c is not None: send_dx_dy(0,0,i2c)
  i2c.close()

if __name__ == '__main__':
  atexit.register(exit_handler)
  main()
