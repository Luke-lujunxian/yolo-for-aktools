import sys
import argparse
from yolo import YOLO
from PIL import Image
import base64


def request(modelPath, anchorsPath, classesPath,OCRmodel,OCRanvhors,OCRclass,image):
    parser = init()
    FLAGS = parser.parse_args({
        '--model': modelPath,
        '--anchors': anchorsPath,
        '--classes': classesPath,
        '--gpu_num': 0,
    })
    FLAGSnum = parser.parse_args({
        '--model': OCRmodel,
        '--anchors': OCRanvhors,
        '--classes': OCRclass,
        '--gpu_num': 0,
    })
    return detect_img(YOLO(**vars(FLAGS)),image,YOLO(**vars(FLAGSnum)))


def detect_img(objReg, image,numReg):
    image = base64.b64decode(image)
    out_boxes, out_scores, out_classes = objReg.detect_image(image)
    objReg.close_session()



FLAGS = None


def init():
    # class YOLO defines the default value, so suppress any default here
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument(
        '--model', type=str,
        help='path to model weight file, default ' + YOLO.get_defaults("model_path")
    )

    parser.add_argument(
        '--anchors', type=str,
        help='path to anchor definitions, default ' + YOLO.get_defaults("anchors_path")
    )

    parser.add_argument(
        '--classes', type=str,
        help='path to class definitions, default ' + YOLO.get_defaults("classes_path")
    )

    parser.add_argument(
        '--gpu_num', type=int,
        help='Number of GPU to use, default ' + str(YOLO.get_defaults("gpu_num"))
    )

    parser.add_argument(
        '--image', default=False, action="store_true",
        help='Image detection mode, will ignore all positional arguments'
    )
    '''
    Command line positional arguments -- for video detection mode
    '''
    parser.add_argument(
        "--input", nargs='?', type=str, required=False, default='./path2your_video',
        help="Video input path"
    )

    parser.add_argument(
        "--output", nargs='?', type=str, default="",
        help="[Optional] Video output path"
    )
    return parser

# The code above and yolo.py are modified from original code writtened by @qqwweee
# https://github.com/qqwweee/keras-yolo3
# The Original Code is using MIT license,
# DO NOT remove this statement
