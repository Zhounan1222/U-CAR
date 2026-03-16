
import os
import json
import re
import numpy as np
from PIL import Image
import torch.utils.data as data
from transformers import BertTokenizer, AutoImageProcessor,ViTFeatureExtractor
import torch
import ijson


class FieldParser:
    def __init__(
            self,
            args
    ):
        super().__init__()
        self.args = args
        self.dataset = args.dataset
        self.vit_feature_extractor = AutoImageProcessor.from_pretrained(args.vision_model)
        


    def _parse_image(self, img):
        pixel_values = self.vit_feature_extractor(img, return_tensors="pt").pixel_values
        return pixel_values[0] 

    # from https://github.com/cuhksz-nlp/R2Gen/blob/main/modules/tokenizers.py
    def clean_report(self, report):
        # clean Iu-xray reports
        if self.dataset == "iu_xray":
            report_cleaner = lambda t: t.replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '') \
            .replace('. 2. ', '. ').replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ') \
            .replace(' 2. ', '. ').replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
            .strip().lower().split('. ')
            
            sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').
                                            replace('\\', '').replace("'", '').strip().lower())
            tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
            report = ' . '.join(tokens) + ' .'
        # clean MIMIC-CXR reports
        else:
            report_cleaner = lambda t: t.replace('\n', ' ').replace('__', '_').replace('__', '_').replace('__', '_') \
                .replace('__', '_').replace('__', '_').replace('__', '_').replace('__', '_').replace('  ', ' ') \
                .replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ') \
                .replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.') \
                .replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '').replace('. 2. ', '. ') \
                .replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ').replace(' 2. ', '. ') \
                .replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ').replace(':', ' :') \
                .strip().lower().split('. ')
            sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+()\[\]{}]', '', t.replace('"', '').replace('/', '')
                                .replace('\\', '').replace("'", '').strip().lower())
            tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
            report = ' . '.join(tokens) + ' .' 
        # report = ' '.join(report.split()[:self.args.max_txt_len])
        return report


    def parse(self, features):
        
        to_return = {'id': features['id']}
        report = features.get("report", "")
        report = self.clean_report(report)
        to_return['input_text'] = report
        to_return['label'] =  features['label']
        
      
        similar_image_feature_list = []
        single_similar_image_topk = []

        for item in features['top-k similar image path']:
            images_similar = []
            
            image_path =item
            for i in range (len(image_path)):
                image_path_single = image_path[i]
                with Image.open(os.path.join('./images/', image_path_single)) as pil:
                    array = np.array(pil, dtype=np.uint8)
                    if array.shape[-1] != 3 or len(array.shape) != 3:
                        array = np.array(pil.convert("RGB"), dtype=np.uint8)
                    image = self._parse_image(array)
                    images_similar.append(image)
            images_similar = torch.stack(images_similar)
                
            similar_image_feature_list.append(images_similar)
        similar_image_feature_list = torch.stack(similar_image_feature_list)
        
        single_similar_image_topk.append(similar_image_feature_list) 
        
        to_return["similar_images_path"] = single_similar_image_topk
        
        images = []
        with Image.open(os.path.join(self.args.base_dir, features['image_path'])) as pil:
            array = np.array(pil, dtype=np.uint8)
            if array.shape[-1] != 3 or len(array.shape) != 3:
                array = np.array(pil.convert("RGB"), dtype=np.uint8)
            image = self._parse_image(array)
            images.append(image)
        to_return["image"] = images
        with open('./image_to_report_similar.json', 'r',encoding="utf-8") as f:

            similar_report= json.load(f)
    
        results_similar_report = []
        if features['id'][0] in similar_report:
            results = similar_report[features['id']]
            for i in range(len(results)):
                results_similar_report.append(results[i]['report'])
        to_return['similar_report'] = self.clean_report(results_similar_report)


        return to_return

    # def transform_with_parse(self, inputs,clip_index):
        
        # return self.parse(inputs,clip_index)
    def transform_with_parse(self, inputs):
        return self.parse(inputs)
     
    def parse_u(self, features):
        to_return = {'id': features['id']}
        report = features.get("report", "")
        report = self.clean_report(report)
        to_return['input_text'] = report
        to_return['label'] =  features['label']
        similar_image_feature_list = []
        single_similar_image_topk = []
        report_similar =[]
        
        
        
        
        for item in features['top-k report to image similar']:
            images_similar = []
            
            image_path = item.get("image_path", "")
            for i in range (len(image_path)):
                image_path_single = image_path[i][0]
                with Image.open(os.path.join(self.args.base_dir, image_path_single)) as pil:
                    array = np.array(pil, dtype=np.uint8)
                    if array.shape[-1] != 3 or len(array.shape) != 3:
                        array = np.array(pil.convert("RGB"), dtype=np.uint8)
                    image = self._parse_image(array)
                    images_similar.append(image)
            images_similar = torch.stack(images_similar)
                
            similar_image_feature_list.append(images_similar)
        similar_image_feature_list = torch.stack(similar_image_feature_list)
        
        single_similar_image_topk.append(similar_image_feature_list) 
        
        to_return["similar_images_path"] = single_similar_image_topk
        for j in range (len(features['similar report feature'])):
            
            report_similar.append(torch.Tensor(features['similar report feature'][j]))
        report_similar_single = torch.stack(report_similar)
        to_return['similar report feature'] = report_similar_single
        

        
        images = []
        for image_path in features['image_path']:
            with Image.open(os.path.join(self.args.base_dir, image_path[0])) as pil:
                array = np.array(pil, dtype=np.uint8)
                if array.shape[-1] != 3 or len(array.shape) != 3:
                    array = np.array(pil.convert("RGB"), dtype=np.uint8)
                image = self._parse_image(array)
                images.append(image)
        to_return["image"] = images
        results_similar_report = []
        with open('./image_to_report_similar.json', 'r',encoding="utf-8") as f:

            similar_report= json.load(f)
    
        if features['id'] in similar_report:
            # results_similar_report = []
            results = similar_report[features['id']]
            for i in range(len(results)):
                results_similar_report.append(results[i]['report'])
        to_return['similar_report'] = results_similar_report
        if self.args.drop_json !='None':
            to_return['answer'] = features["answer"][0][0]
            to_return['entropy'] = features["entropy"]


        return to_return

    # def transform_with_parse(self, inputs,clip_index):
        
        # return self.parse(inputs,clip_index)
    def transform_with_parse(self, inputs):
        
        return self.parse(inputs)
    def transform_with_parse_positive(self, inputs):
        return self.parse_positive(inputs)

class ParseDataset(data.Dataset):
    def __init__(self, args, split='train'):
        self.args = args
        
        self.meta = json.load(open(args.annotation, 'r'))
        self.meta = self.meta[split]
        
        # # 如果是重新生成阶段：过滤掉不在 drop.json 中的样本
        if args.drop_json !='None':  
            with open(args.drop_json, 'r') as f:
                dropped_ids = set(json.load(f))

            
            
            if args.dataset =='iu_xray':
             
                with open('iuxray_ref_dropped_0.3.json', 'r') as f:
                    answer_dict = json.load(f)  # e.g., {id: "generated report text"}

                with open('iuxray_ref_dropped_0.3_uncertainty.json', 'r') as f:
                    entropy_dict = json.load(f)  # e.g., {id: [e1, e2, ..., eT]}
            else:
                with open('mimic_ref_dropped_0.3.json', 'r') as f:
                    answer_dict = json.load(f)  # e.g., {id: "generated report text"}

                with open('mimic_ref_dropped_0.3_uncertainty.json', 'r') as f:
                    entropy_dict = json.load(f)  # e.g., {id: [e1, e2, ..., eT]}


            # === 过滤 + 注入补充字段 ===
            split_meta = []
            for sample in self.meta:
                sample_id = sample["id"][0]
                if sample_id in dropped_ids:
                    sample["answer"] = answer_dict.get(sample_id, "")
                    sample["entropy"] = entropy_dict.get(sample_id, [])
                    split_meta.append(sample)

            # 保存到 self.meta
            self.meta = split_meta
        
        self.parser = FieldParser(args)
        
    def load_split_json(self,path, split_name):
        with open(path, 'r') as f:
            parser = ijson.kvitems(f, '')
            for key, value in parser:
                if key == split_name:
                    return value  # 直接返回指定 split 的内容（例如 train: [ {...}, {...}, ... ]）
        return []


    def __len__(self):
        return len(self.meta)
   

    def __getitem__(self, index):
        return self.parser.transform_with_parse(self.meta[index])
    
    
    
from torchvision import transforms

def create_datasets(args):
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    dev_dataset = ParseDataset(args, 'val')
    test_dataset = ParseDataset(args, 'test')
    train_dataset = ParseDataset(args, 'train')
    return train_dataset, dev_dataset, test_dataset


