# tf2_to_tflite_conversion:

## (1) Install tensorflow dependency

pip install tensorflow-gpu==2.x or 1.x

## (2) Install tensorflow object detection api dependency

```
cd models/research/
protoc object_detection/protos/*.proto --python_out=.
cp object_detection/packages/tf1/setup.py .
python -m pip install .
```
## (3) Run the validation tests

```
python /content/models/research/object_detection/builders/model_builder_tf1_test.py
```
## (4) Download an object detection model (from zoo or trained)

```
wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03.tar.gz
tar -xvf ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03.tar.gz
```
```
ls -lah ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03/
```
## (5) Export tflite graph (.pb)

```
python models/research/object_detection/export_tflite_ssd_graph.py \
    --pipeline_config_path ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03/pipeline.config \
    --trained_checkpoint_prefix ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03/model.ckpt \
    --output_directory tflite/ \
    --add_postprocessing_op=true
```
``` 
ls -lah tflite
```
## (6) Export tflite conversion file (.tflite)

```
tflite_convert \
  --output_file tflite/tflite_graph.tflite \
  --graph_def_file tflite/tflite_graph.pb \
  --inference_type QUANTIZED_UINT8 \
  --input_arrays normalized_input_image_tensor \
  --output_arrays TFLite_Detection_PostProcess,TFLite_Detection_PostProcess:1,TFLite_Detection_PostProcess:2,TFLite_Detection_PostProcess:3 \
  --mean_values 128 \
  --std_dev_values 128 \
  --input_shapes 1,300,300,3 \
  --change_concat_input_ranges false \
  --allow_nudging_weights_to_use_fast_gemm_kernel true \
  --allow_custom_ops
```
```
ls -lah tflite
```

## (7) Run inference on test data
```
python tflite_detection.py
```
