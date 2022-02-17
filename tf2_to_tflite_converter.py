import argparse
from inspect import signature
import cv2
import imageio
import numpy as np
import os
import time
import tensorflow as tf

TEST_DATA = 'test_data/test_one_car_image.jpeg'

def tf2_to_tflite(config):
    tf_input_path = config['tf2_input_path']
    tflite_output_path = config['tflite_output_path']
    in_height, in_width, in_chs = config['input_shape']
    tf_model_name = tf_input_path.split('saved_model')[0].split(os.path.sep)[:-1][-1]

    pretrained_model = tf.saved_model.load(tf_input_path)
    if __debug__:
        print(f'Signatures keys: {list(pretrained_model.signatures.keys())}\n')

    signatures_def = pretrained_model.signatures["serving_default"]

    signature_keys = []
    for k, v in signatures_def.structured_input_signature[-1].items():
        print(f'{k} : {v}\n')
        signature_keys.append(k)

    for k, v in signatures_def.structured_outputs.items():
        print(f'{k} : {v}\n')
        signature_keys.append(k)

    print(signature_keys)

    converter = tf.lite.TFLiteConverter.from_saved_model(
        tf_input_path,
        signature_keys=['serving_default']
    )

    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    tflite_model = converter.convert()

    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    signatures = interpreter.get_signature_list()
    print(f'List of signatures (after conversion): {signatures}')

    print(f'\nInputs:\n{interpreter.get_input_details()}\n')
    print(f'\nOutputs:\n{interpreter.get_output_details()}\n')

    with open(os.path.join(tflite_output_path, f'{tf_model_name}.tflite'), 'wb') as file:
        file.write(tflite_model)

def load_saved_model(saved_model_dir):
    loaded_model = tf.saved_model.load(saved_model_dir)
    print(f'Signatures keys: {list(loaded_model.signatures.keys())}\n')

    inference = loaded_model.signatures["serving_default"]
    for k, v in inference.structured_outputs.items():
        print(f'{k} : {v}\n')

    img = imageio.imread(TEST_DATA).astype(np.uint8)
    img = cv2.resize(img, (320, 320))
    img = tf.convert_to_tensor(img[np.newaxis, ...], dtype=tf.uint8)
    outputs = inference(img)
    num_detections = outputs['num_detections']
    detection_boxes = outputs['detection_boxes']
    detection_classes = outputs['detection_classes']
    detection_scores = outputs['detection_scores']

def test_tflite_conversion_without_signature(tflite_model_path):
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    dummy_img = np.zeros((1, 320, 320, 3)).astype(np.uint8)
    interpreter.set_tensor(input_details[0]['index'], dummy_img)

    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print(output_data)

def test_tflite_conversion(tflite_model_path):
    interpreter = tf.lite.Interpreter(tflite_model_path)
    # There is only 1 signature defined in the model,
    # so it will return it by default.
    # If there are multiple signatures then we can pass the name.
    my_signature = interpreter.get_signature_runner()

    # my_signature is callable with input as arguments.
    dummy_img = np.zeros((1, 320, 320, 3)).astype(np.uint8)
    output = my_signature(x=dummy_img, shape=(1, 320, 320, 3), dtype=tf.uint8)
    # 'output' is dictionary with all outputs from the inference.
    # In this case we have single output 'result'.
    print(output['result'])

# Note: Converts from SavedModel (recommended way)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str)
    parser.add_argument('--tf_input_path', type=str)
    parser.add_argument('--tflite_output_path', type=str)
    parser.add_argument('--tflite_model_path', type=str)

    args = parser.parse_args()

    if args.mode == 'convert':
        print('Conversion has started')

        #load_saved_model(args.tf_input_path,)

        config = {
            'tf2_input_path' : args.tf_input_path,
            'tflite_output_path' : args.tflite_output_path,
            # HWC
            'input_shape' : (320, 320, 3)
        }

        st = time.time()
        tf2_to_tflite(config)
        print(f'time: {(time.time() - st)}s')

        print('Conversion is complete')
    elif args.mode == 'test':
        test_tflite_conversion_without_signature(args.tflite_model_path)
