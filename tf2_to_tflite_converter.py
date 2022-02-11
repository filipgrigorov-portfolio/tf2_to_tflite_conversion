import argparse
import os
import time
import tensorflow as tf

def tf2_to_tflite(saved_model_dir, output_dir):
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    tflite_model = converter.convert()

    with open(os.path.join(output_dir, 'tflite_od.tflite'), 'wb') as file:
        file.write(tflite_model)

# Note: Converts from SavedModel (recommended way)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--saved_model_dir', type=str)
    parser.add_argument('--output_dir', type=str)

    args = parser.parse_args()

    print('Conversion has started')

    st = time.time()
    tf2_to_tflite(args.saved_model_dir, args.output_dir)
    print(f'time: {(time.time() - st) * 1e6}us')

    print('Conversion is complete')
