from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer, ColorMode
import matplotlib.pyplot as plt
import cv2
import os
from detectron2.data.datasets import register_coco_instances

from sahi.utils.detectron2 import Detectron2TestConstants

# import required functions, classes
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction, predict, get_prediction
from sahi.utils.file import download_from_url
from sahi.utils.cv import read_image
from IPython.display import Image

def Inference_on_0909():
    files_path = r"/scratch/tjian/PythonProject/deep_plastic_Flux_SSL/checkpoint/train_weights/SL/20per_10_runs/"
    files = os.listdir(files_path)
    model_config_path = "/scratch/tjian/PythonProject/deep_plastic_Flux_SSL/checkpoint/train_weights/SL/SL_20per_1/config.yaml"

    source_image_dir = r"/scratch/tjian/Data/Flux/TUD_Vietnam/0909/images_no_litter/"  # 0909
    # source_image_dir = r"/scratch/tjian/Data/Flux/TUD_Vietnam/1209/images_no_litter/"  # 1209

    ori_output_dir = r"/scratch/tjian/Data/Flux/TUD_Vietnam/0909/Pred/SL_20per/SA_1280_Conf_0.9/No_litter"  # 0909
    # ori_output_dir = r"/scratch/tjian/Data/Flux/TUD_Vietnam/1209/Pred/SSL_20per/SA_1920_Conf_0.9/No_litter"  # 1209

    # # Flux_train_20per
    register_coco_instances("Flux_train_20per", {}, "/scratch/tjian/Data/Flux/labeled_images_new/annotations/Train_20per.json", "/scratch/tjian/Data/Flux/labeled_images_new/Train_20per/")
    register_coco_instances("Flux_val_20per", {}, "/scratch/tjian/Data/Flux/labeled_images_new/annotations/Val_20per.json", "/scratch/tjian/Data/Flux/labeled_images_new/Val_20per/")
    Train_Dataset_name="Flux_train_20per"
    Test_Dataset_name="Flux_val_20per"
    Class_name=["litter"]
    MetadataCatalog.get(Train_Dataset_name).set(thing_classes=Class_name)
    MetadataCatalog.get(Test_Dataset_name).set(thing_classes=Class_name)

    for filename in files:
        print("#################", filename, "################")
        head, sep, tail = filename.partition('.pth')
        if str(sep) == '.pth':
          new_folder_name = head  # build the new name
        model_checkpoint_path = os.path.join(files_path, filename)
        detection_model = AutoDetectionModel.from_pretrained(
          model_type='detectron2',
          model_path=model_checkpoint_path,
          config_path=model_config_path,
          confidence_threshold=0.9,
          # image_size=1333,
          device="cuda:0",
          )
        # do not generate the output_dir, the code will do it automatically
        output_dir = os.path.join(ori_output_dir, new_folder_name)
        # define slicing conf.
        slice_height = 1280
        slice_width = 1280
        overlap_height_ratio = 0.2
        overlap_width_ratio = 0.2
        result = predict(
          detection_model = detection_model,
          source = source_image_dir,
          slice_height = slice_height,
          slice_width = slice_width,
          overlap_height_ratio = overlap_height_ratio,
          overlap_width_ratio = overlap_width_ratio,
          project = output_dir,
          postprocess_type = 'NMS', # dDefault is 'GREEDYNMM'.
          postprocess_match_metric = 'IOU',
          postprocess_match_threshold = 0.5, # NMS_IoU_threshold
          visual_text_size = 3 # float
          )


if __name__ == "__main__":
    Inference_on_0909()
    
