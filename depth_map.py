import time
import PIL.Image as pil
import numpy as np

import mxnet as mx
from mxnet.gluon.data.vision import transforms

import gluoncv
from gluoncv.model_zoo.monodepthv2.layers import disp_to_depth

import cv2

def gen_depthmap(img):
    # using cpu
    ctx = mx.cpu(0)

    # convert openCV2 to PIL
    color_coverted = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = pil.fromarray(color_coverted)

    original_width, original_height = img.size

    model_zoo = 'monodepth2_resnet18_kitti_mono_stereo_640x192'
    model = gluoncv.model_zoo.get_model(model_zoo, pretrained_base=False, ctx=ctx, pretrained=True)

    min_depth = 0.1
    max_depth = 100

    # while use stereo or mono+stereo model, we could get real depth value
    scale_factor = 5.4
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80

    feed_height = 192
    feed_width = 640

    img = img.resize((feed_width, feed_height), pil.LANCZOS)
    img = transforms.ToTensor()(mx.nd.array(img)).expand_dims(0).as_in_context(context=ctx)

    outputs = model.predict(img)
    mx.nd.waitall()
    pred_disp, _ = disp_to_depth(outputs[("disp", 0)], min_depth, max_depth)
    t = time.time()
    pred_disp = pred_disp.squeeze().as_in_context(mx.cpu()).asnumpy()
    pred_disp = cv2.resize(src=pred_disp, dsize=(original_width, original_height))

    pred_depth = 1 / pred_disp
    pred_depth *= scale_factor
    pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
    pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH
    dimg = (pred_depth)

    return dimg

    
if __name__ == '__main__':
    im = cv2.imread('img.jpg')
    dim = gen_depthmap(im)

    cv2.imwrite('dimg.jpg', dim)