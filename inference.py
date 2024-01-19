# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Copyright (c) Institute of Information Processing, Leibniz University Hannover.
# Copyright (c) Paolo Dimasi, Politecnico di Torino.
import argparse, json, os
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import torchvision.transforms as T
from models import build_model

from utils.common import get_args_parser

def main(args):

    transform = T.Compose([
        T.Resize(800),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    # for output bounding box post-processing
    def box_cxcywh_to_xyxy(x):
        x_c, y_c, w, h = x.unbind(1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
             (x_c + 0.5 * w), (y_c + 0.5 * h)]
        
        return torch.stack(b, dim=1)

    def rescale_bboxes(out_bbox, size):
        img_w, img_h = size
        b = box_cxcywh_to_xyxy(out_bbox)
        b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
        return b

    # TG classes
    with open('traffic_genome_format/traffic_genome_format.json') as f:
        tmp = json.load(f)
        CLASSES = list(tmp['idx_to_label'].values())

        REL_CLASSES = list(tmp['idx_to_predicate'].values())

    model, _, _ = build_model(args)
 
    checkpoint = torch.load(args.resume, map_location='cpu')
    model.load_state_dict(checkpoint['model'], strict=True)
    model.eval()



    img_path = args.img_path
    tmp = Path(img_path).stem

    if args.enable_ltn:
        postfix = '_ltn.jpg' 
        
    elif args.enable_kg:
        postfix = '_kge.jpg'   
        
    else:
        postfix = '.jpg'
        
    inf_path= f'{args.inf_dir}/{os.path.basename(tmp)}_inf{postfix}'
    
    im = Image.open(img_path)

    # mean-std normalize the input image (batch-size: 1)
    img = transform(im).unsqueeze(0)

    # propagate through the model
    outputs = model(img)

    # keep only predictions with 0.+ confidence
    probas = outputs['rel_logits'].softmax(-1)[0, :, :-1]
    probas_sub = outputs['sub_logits'].softmax(-1)[0, :, :-1]
    probas_obj = outputs['obj_logits'].softmax(-1)[0, :, :-1]
    keep = torch.logical_and(probas.max(-1).values > args.conf, torch.logical_and(probas_sub.max(-1).values >args.conf,
                                                                            probas_obj.max(-1).values > args.conf))

    # convert boxes from [0; 1] to image scales
    sub_bboxes_scaled = rescale_bboxes(outputs['sub_boxes'][0, keep], im.size)
    obj_bboxes_scaled = rescale_bboxes(outputs['obj_boxes'][0, keep], im.size)

    topk = 30
    keep_queries = torch.nonzero(keep, as_tuple=True)[0]
    indices = torch.argsort(-probas[keep_queries].max(-1)[0] * probas_sub[keep_queries].max(-1)[0] * probas_obj[keep_queries].max(-1)[0])[:topk]
    keep_queries = keep_queries[indices]

    # use lists to store the outputs via up-values
    conv_features, dec_attn_weights_sub, dec_attn_weights_obj = [], [], []

    hooks = [
        model.backbone[-2].register_forward_hook(
            lambda self, input, output: conv_features.append(output)
        ),
        model.transformer.decoder.layers[-1].cross_attn_sub.register_forward_hook(
            lambda self, input, output: dec_attn_weights_sub.append(output[1])
        ),
        model.transformer.decoder.layers[-1].cross_attn_obj.register_forward_hook(
            lambda self, input, output: dec_attn_weights_obj.append(output[1])
        )
    ]
    with torch.no_grad():
        # propagate through the model
        outputs = model(img)

        for hook in hooks:
            hook.remove()

        # don't need the list anymore
        conv_features = conv_features[0]
        dec_attn_weights_sub = dec_attn_weights_sub[0]
        dec_attn_weights_obj = dec_attn_weights_obj[0]

        # get the feature map shape
        h, w = conv_features['0'].tensors.shape[-2:]
        im_w, im_h = im.size

        fig, axs = plt.subplots(ncols=len(indices), nrows=3, figsize=(22, 7))
        for idx, ax_i, (sxmin, symin, sxmax, symax), (oxmin, oymin, oxmax, oymax) in \
                zip(keep_queries, axs.T, sub_bboxes_scaled[indices], obj_bboxes_scaled[indices]):
            ax = ax_i[0]
            ax.imshow(dec_attn_weights_sub[0, idx].view(h, w))
            ax.axis('off')
            ax.set_title(f'query id: {idx.item()}')
            ax = ax_i[1]
            ax.imshow(dec_attn_weights_obj[0, idx].view(h, w))
            ax.axis('off')
            ax = ax_i[2]
            ax.imshow(im)
            ax.add_patch(plt.Rectangle((sxmin, symin), sxmax - sxmin, symax - symin,
                                       fill=False, color='blue', linewidth=2.5))
            ax.add_patch(plt.Rectangle((oxmin, oymin), oxmax - oxmin, oymax - oymin,
                                       fill=False, color='orange', linewidth=2.5))

            ax.axis('off')
            ax.set_title(CLASSES[probas_sub[idx].argmax()]+' '+REL_CLASSES[probas[idx].argmax()]+' '+CLASSES[probas_obj[idx].argmax()], fontsize=10)

        fig.tight_layout()
        
        plt.savefig(inf_path, dpi=300, bbox_inches='tight')
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser('RelTR inference', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
