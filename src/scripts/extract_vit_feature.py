import cv2
import math
import timm
import torch
import pickle
import argparse
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torchvision import transforms
from timm.models import VisionTransformer

import sys
sys.path.append('src')
from model import ALBEF
from dataset import MatterSimEnv
from config import CANDIDATE_BUFFER_PATH, ALBEF_PATH
import utils


class Extractor:
    def __init__(self):
        # self.env = MatterSimEnv(True, (518, 518), vfov=60)
        # self.env = MatterSimEnv(True, (384, 384), vfov=60)
        self.env = MatterSimEnv(True, (224, 224), vfov=60)

        with open(CANDIDATE_BUFFER_PATH, 'rb') as f:
            print('Loading candidate buffer...')
            self.buffer = pickle.load(f)
        
        self.device = torch.device('cuda:3')

        # ALBEF
        # print('loading ALBEF...')
        # albef = ALBEF()
        # checkpoint = torch.load(ALBEF_PATH)
        # albef.load_state_dict(checkpoint)
        # self.vit = albef.visual_encoder.to(self.device)
        # self.transform = albef.get_transform()
        
        # CLIP-vit-base-patch16
        from transformers import CLIPProcessor, CLIPModel
        path = '/home/lurenjie/documents/pretrained/clip-vit-base-patch16'
        self.processor = CLIPProcessor.from_pretrained(path)
        self.transform = lambda imgs: self.processor(images=imgs, return_tensors='pt').pixel_values[0]
        self.vit = CLIPModel.from_pretrained(path).vision_model.to(self.device)

        # DINOv2
        # 'facebookresearch/dinov2'
        # self.vit = torch.hub.load(
        #     '/home/lurenjie/.cache/torch/hub/facebookresearch_dinov2_main', 'dinov2_vitb14',
        #     source='local').to(self.device)
        # self.transform = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        # ])

        self.vit.eval()

    @torch.no_grad()
    def extract_cls_features(self, args):
        with open('./connectivity/scans.txt') as f:
            scans = f.readlines()

        with open(args.output_path, 'wb') as f:
            # check path exists
            pass

        features = dict()
        for i, scan_id in enumerate(scans):
            scan_id = scan_id.strip()
            print(i+1, scan_id)

            graph = utils.load_graph(scan_id)
            for viewpoint_id in tqdm(graph.nodes):
                imgs = []
                k = 12
                for view_index in range(k * 3):
                    heading = (view_index % k) * math.radians(360 / k)
                    elevation = (view_index // k - 1) * math.radians(360 / k)

                    self.env.new_episode(scan_id, viewpoint_id, heading, elevation)
                    state = self.env.get_state()
                    rgb = utils.to_numpy(state.rgb)

                    img = self.transform(rgb).to(self.device)  # (3, H, W)
                    imgs.append(img)
                imgs = torch.stack(imgs, dim=0)

                long_id = '%s_%s' % (scan_id, viewpoint_id)
                # features[long_id] = self.vit(imgs).cpu().numpy()  # (36, 768)
                # features[long_id] = self.vit(imgs).pooler_output.cpu().numpy()  # (36, 768)
                features[long_id] = self.vit(imgs).last_hidden_state[:, 0, :].cpu().numpy()  # (36, 768)
        with open(args.output_path, 'wb') as f:
            pickle.dump(features, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type=str, required=True)
    args = parser.parse_args()
    Extractor().extract_cls_features(args)
