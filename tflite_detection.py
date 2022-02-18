import copy
import numpy as np
import sys

import tensorflow as tf

from PIL import Image, ImageDraw, ImageFont
from six import BytesIO

def set_input_tensor(interpreter, img):
    # [{
    #   'name': 'normalized_input_image_tensor', 'index': 260, 'shape': array([  1, 300, 300,   3], dtype=int32), 
    #   'dtype': <class 'numpy.uint8'>, 'quantization': (0.0078125, 128)
    # }]
    tensor_idx = interpreter.get_input_details()[0]['index']
    # BHWC -> HWC
    input_tensor = interpreter.tensor(tensor_idx)()[0]
    # deep copy the content of PIL.Image
    # (300, 300, 3)
    input_tensor[...] = copy.deepcopy(img)
    # clear any numpy array referencing inter buffers

def get_output_tensor(interpreter, idx):
    '''
        [
            {'name': 'TFLite_Detection_PostProcess', 'index': 252, 'shape': array([ 1, 10,  4], 
        dtype=int32), 'dtype': <class 'numpy.float32'>, 'quantization': (0.0, 0)}, 
            
            {'name': 'TFLite_Detection_PostProcess:1', 'index': 253, 'shape': array([ 1, 10], dtype=int32), 
        'dtype': <class 'numpy.float32'>, 'quantization': (0.0, 0)}, 
            
            {'name': 'TFLite_Detection_PostProcess:2', 'index': 254, 'shape': array([ 1, 10], dtype=int32), 
        'dtype': <class 'numpy.float32'>, 'quantization': (0.0, 0)}, 
            
            {'name': 'TFLite_Detection_PostProcess:3', 'index': 255, 'shape': array([1], dtype=int32), 
        'dtype': <class 'numpy.float32'>, 'quantization': (0.0, 0)}]
    '''
    output_list = interpreter.get_output_details()
    output_tensor = np.squeeze(interpreter.tensor(idx)())
    return output_tensor

def detect(interpreter, img):
    # allocate img to input tensor
    set_input_tensor(interpreter, img)
    # invoke the interpreter
    interpreter.invoke()
    # get outputs from output tensor
    num_dets = get_output_tensor(interpreter, 255)
    bboxes = get_output_tensor(interpreter, 252)
    classes = get_output_tensor(interpreter, 253)
    scores = get_output_tensor(interpreter, 254)
    # arrange into data struct
    return int(num_dets), bboxes, classes, scores

def filterby(num_dets, bboxes, classes, scores, conf_thresh=0.75):
    '''indices = []
    for idx in range(num_dets):
        if scores[idx] >= conf_thresh:
            indices.append(idx)'''
    indices = scores >= conf_thresh
    bboxes = bboxes[indices]
    classes = classes[indices]
    scores = scores[indices]
    return len(indices[indices == True]), bboxes, classes, scores

def draw_bboxes(img, num_dets, bboxes, classes, scores):
    img_size = img.size
    canvas = ImageDraw.Draw(img)

    for idx in range(num_dets):
        xmin, ymin, xmax, ymax = bboxes[idx]
        xmin *= img_size[1]
        ymin *= img_size[0]
        xmax *= img_size[1]
        ymax *= img_size[0]
        lbl = classes[idx]
        score = scores[idx]

        canvas.rectangle([ymin, xmin, ymax, xmax], outline=(255, 0, 0))
        canvas.text((ymin + int(0.3 * (ymax - ymin)), xmin + int(0.2 * (xmax - xmin))), str(lbl), (255, 0, 0, 0))
        canvas.text((ymin + int(0.3 * (ymax - ymin)), xmin + int(0.22 * (xmax - xmin))), str(score), (255, 0, 0, 0))

    img.show()

if __name__ == '__main__':
    interpreter = tf.lite.Interpreter(model_path="tflite_models/ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03.tflite")
    interpreter.allocate_tensors()
    batch, in_h, in_w, chs = interpreter.get_input_details()[0]['shape']
    print(f'input: {batch}x{in_h}x{in_w}x{chs}')

    img = Image.open('test_data/cars.jpg')
    img_in = img.resize((in_w, in_h), Image.ANTIALIAS)
    num_dets, bboxes, classes, scores = detect(interpreter, img_in)
    print(num_dets)
    num_dets, bboxes, classes, scores = filterby(num_dets, bboxes, classes, scores, 0.5)
    print(num_dets)
    draw_bboxes(img, num_dets, bboxes, classes, scores)
