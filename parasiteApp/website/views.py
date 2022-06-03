
from .models import Parasite, ParasiteManager
from .forms import Parasiteform
from django.http import HttpResponseRedirect, FileResponse
from django.shortcuts import render, redirect

import numpy as np
import cv2
from distutils.version import StrictVersion
from PIL import Image, ImageDraw
import sys
from io import BytesIO
from django.core.files.storage import FileSystemStorage
from django.core.files.uploadedfile import InMemoryUploadedFile

from django.core.files.base import ContentFile

import json

from tensorflow_serving.apis.predict_pb2 import PredictRequest
from tensorflow_serving.apis import prediction_service_pb2_grpc
from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
import keras as keras #antes era tensorflow.keras
import tensorflow as tf

import matplotlib.pyplot as plt
import grpc
import matplotlib

matplotlib.use('TkAgg')

# Create your views here.

#SERVER = '3.128.144.86:8500'



def home(request):
    
    if request.method == "POST":
      parasite_form = Parasiteform(request.POST, request.FILES)

      if parasite_form.is_valid(): #si el form cumple con los requsiitos del template
        parasite_form.save() #lo guardamos en la database
        return redirect('showImage')

    else:
        parasite_form = Parasiteform()
       
    return render(request, 'home.html', {'form': parasite_form})

def showImage(request):
    
    parasite = Parasite.objects.last() #obtenemos la ultima imagen guardada
    
    image_as_array = np.array(Image.open(parasite.image))

    tested_image_as_array = make_detection(image_as_array)
    detected_pil = Image.fromarray(np.uint8(tested_image_as_array))
    image_uri = to_data_uri(detected_pil)
    
    return render(request, 'showImage.html', {"image_uri": image_uri})
    


def make_detection(image_np):

    import os
    
    myjsonfile = open('website/json_files/configuration_data.json', 'r')
    jsondata = myjsonfile.read()
    object = json.loads(jsondata)

    SERVER = str(object['server'])
    MAP = str(object['map'])


    request = PredictRequest()
    request.model_spec.name = "parasite_model"
    request.model_spec.signature_name = "serving_default"
   
    request.inputs['input_tensor'].CopyFrom(tf.make_tensor_proto(image_np[np.newaxis, :, :, :]))

    channel = grpc.insecure_channel(
        SERVER,
        options=[('grpc.max_send_message_length', -1),
             ('grpc.max_receive_message_length', -1)]
    )
    predict_service = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    response = predict_service.Predict(request, timeout=60)

    num_detections = int(tf.make_ndarray(response.outputs["num_detections"])[0])
    output_dict = {
        'detection_boxes': tf.make_ndarray(response.outputs["detection_boxes"]),
        'detection_classes': tf.make_ndarray(response.outputs["detection_classes"]).astype('int64'),
        'detection_scores': tf.make_ndarray(response.outputs["detection_scores"])
    }
    output_dict = {key: value[0, :num_detections] for key, value in output_dict.items()}
    output_dict['num_detections'] = num_detections

    if 'detection_masks' in response.outputs:
      # Reframe the the bbox mask to the image size.
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            output_dict['detection_masks'], output_dict['detection_boxes'],
            image_np.shape[0], image_np.shape[1])
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5, tf.uint8)
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()


    category_index = label_map_util.create_category_index_from_labelmap(MAP,
                                                            use_display_name=True)
    annotated_img = vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks_reframed', None),
        use_normalized_coordinates=True,
        line_thickness=8
    )
   
  
    return image_np

def to_image(numpy_img):
    img = Image.fromarray(numpy_img, 'RGB')
    return img

import base64
from io import BytesIO
def to_data_uri(pil_img):
    data = BytesIO()
    pil_img.save(data, "png") # pick your format
    data64 = base64.b64encode(data.getvalue())
    return u'data:img/jpeg;base64,'+data64.decode('utf-8') 

