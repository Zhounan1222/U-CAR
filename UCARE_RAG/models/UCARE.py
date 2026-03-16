import os
import json
import torch
import torch.nn as nn
import lightning.pytorch as pl
from transformers import LlamaForCausalLM, LlamaTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
# from pycocoevalcap.meteor.meteor import Meteor

from transformers import SwinModel
from lightning_tools.optim import config_optimizer
from peft import get_peft_model, LoraConfig, TaskType
import pdb
from models.transformer import Transformer
from einops import rearrange,reduce
import numpy as np
import torch.nn.functional as F
# from hopfield_layers.hflayers import HopfieldLayer
from pytorch_lightning.callbacks import ModelCheckpoint
import ijson
from tqdm import tqdm
class UCARE(pl.LightningModule):
    """
    UCARE model.
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.save_hyperparameters(args)

        print(f'Loading vision encoder:{args.vision_model}')
        self.visual_encoder = SwinModel.from_pretrained(args.vision_model)
        if args.vis_use_lora:
            peft_config_visual = LoraConfig(
                                    r=args.vis_r,
                                    lora_alpha=args.vis_alpha,
                                    target_modules=["query", "value"],
                                    lora_dropout=args.lora_dropout,
                                    bias="none",
                                    modules_to_save=["classifier"],
                                )
            self.visual_encoder = get_peft_model(self.visual_encoder, peft_config_visual)
            self.visual_encoder.print_trainable_parameters()
            print('Loading vision encoder with LoRA -- Done')
        finetune = args.vision_model_clip
        if finetune != 'None':
            checkpoint = torch.load(finetune, map_location='cpu')
            print(f"Load arm pre-trained checkpoint from: {finetune}" )
            checkpoint_model = checkpoint['model']
            new_dict = {}
            for k, v in checkpoint_model.items():
                if "visual_encoder." in k:
                    new_dict[k.replace("visual_encoder.", "")] = v
            # load pre-trained model
            self.visual_encoder.load_state_dict(new_dict,strict=False)

        if args.freeze_vm:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            print(f'Loading Frozen vision encoder:{args.vision_model} -- Done')
        else:
            print(f'Loading Trainable vision encoder:{args.vision_model} -- Done')

        print('Loading LLAMA')
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(args.llama_model, use_fast=False)
        self.llama_tokenizer.pad_token_id = 0
        if args.low_resource:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                args.llama_model,
                torch_dtype=torch.float16,
                load_in_8bit=True,
                device_map="auto"
            )
        else:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                args.llama_model,
                torch_dtype=torch.float16,
            )
         
        if args.llm_use_lora:
            self.embed_tokens = self.llama_model.get_input_embeddings()
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM, inference_mode=False, r=args.llm_r, lora_alpha=args.llm_alpha, lora_dropout=args.lora_dropout,target_modules=["q_proj","v_proj"]
            )
            self.llama_model = get_peft_model(self.llama_model, peft_config)
            self.llama_model.print_trainable_parameters()
            
            llama_dict = {}
            # train
            if finetune != 'None':
                checkpoint = torch.load(finetune, map_location='cpu')
                print(f"Load arm pre-trained checkpoint from: {finetune}" )
                checkpoint_model = checkpoint['model'] 
                for k, v in checkpoint_model.items():
                    if "text_encoder." in k:
                        llama_dict[k.replace("text_encoder.", "")] = v
                self.llama_model.load_state_dict(llama_dict, strict = False)
            
            # 只解冻 LoRA 层的参数
            for name, param in self.llama_model.named_parameters():
                if "lora" in name:  # LoRA 层的名称中一般包含 lora
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            print('Loading LLAMA LoRA Done')       
            # print('Loading LLAMA LoRA Done')         
          
        else:
            self.embed_tokens = self.llama_model.get_input_embeddings()
            for name, param in self.llama_model.named_parameters():
                param.requires_grad = False
            print('Loading LLAMA Done')
            
        self.output_proj = nn.Linear(512, self.visual_encoder.num_features)
        # self.output_proj_report =nn.Linear(512, self.visual_encoder.num_features)
        
        with open("./image_to_report_similarity.json", "r") as f:
            self.image_report_similarity = json.load(f)
        with open("./train_report_features.json", "r") as f:
            self.train_report_feature_dict = json.load(f)
            print('Loading feature Done')

       
        
        self.llama_proj = nn.Linear(self.visual_encoder.num_features, self.llama_model.config.hidden_size)
        self.layer_norm = nn.LayerNorm(self.llama_model.config.hidden_size)
        self.end_sym = args.end_sym
        self.prompt = 'Generate a comprehensive and detailed diagnosis report for this chest xray image.'
        self.prompt_report ='Generate a comprehensive, detailed diagnostic report for this chest X-ray image by referencing similar reports provided.'
        self.val_step_outputs = []
        self.test_step_outputs = []
        self.val_score = 0.0
    
        self.tau = 0.5  # Gumbel-Softmax 温度
        
        self.lambda_report = 1#言型损失权重
        self.lambda_refine = 0.0001     # self-refining loss 权重
        
        self.attention_pooling_similar = ImageAttentionPooling_similar(self.llama_model.config.hidden_size)  # 用注意力计算 image_repr
        
        self.proj_text_similar = nn.Linear(4096, self.llama_model.config.hidden_size)
        
        # train
        
        # temp_weights = self.llama_model.base_model.model.base_model.layers[27].self_attn.q_proj.lora_B.default.weight
        
        if args.delta_file is not None:
            state_dict = torch.load(args.delta_file, map_location=torch.device(f'cuda:{torch.cuda.current_device()}'))['model']
            self.load_state_dict(state_dict=state_dict, strict=False)
            print(f'Load checkpoint from {args.delta_file}')
            
   
    

    def score(self, ref, hypo):
        """
        ref, dictionary of reference sentences (id, sentence)
        hypo, dictionary of hypothesis sentences (id, sentence)
        score, dictionary of scores
        """
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Rouge(), "ROUGE_L"),
            # (Meteor(), "METEOR"),
            (Cider(), "CIDEr")
        ]
        final_scores = {}
        for scorer, method in scorers:
            score, scores = scorer.compute_score(ref, hypo)
            if type(score) == list:
                for m, s in zip(method, score):
                    final_scores[m] = s
            else:
                final_scores[method] = score
        return final_scores
    def get_similar_reports(self, image_ids):
        """
        获取与给定多个 image_ids 相似的 report 特征
        输入：image_ids - 图像 ID 的列表
        输出：返回形状为 [batch_size, num_similar_reports, feature_dim] 的 Tensor
        """
        # 存储每个 image_id 对应的报告特征
        all_report_features = []

        for image_id in image_ids:
            # 获取与该 image_id 相似的报告 ID
            report_features= self.image_report_similarity.get(image_id, []) # 获取相似的报告 ID
       
            all_report_features.append(report_features)

        # 转换为 Tensor，形状为 [batch_size, num_similar_reports, feature_dim]
        return torch.tensor(all_report_features)  # 返回形状 [batch_size, 5, 768]
     
    
    def encode_report_memory(self,image,id):
       
        device = image[0].device
          
        #直接获取相似报告的特征
        selected_report_memory = self.get_similar_reports(id).to(device)  # [B, K, 768]
        report_memory_token =  self.llama_proj(self.output_proj(selected_report_memory)) 
            
        inputs_llama = report_memory_token
        atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(image.device)

        return inputs_llama, atts_llama
    
    
    
    
    def encode_llama_proj(self,image_embeds):
        
        inputs_llama = self.llama_proj(image_embeds)
        atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(image_embeds.device)
        
        return inputs_llama, atts_llama

    
    
    def configure_callbacks(self):
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(self.hparams.savedmodel_path, 'checkpoints'),
            filename="checkpoint_epoch{epochh}_step{step}_{Bleu_4:.6f}_{CIDEr:.6f}",
            monitor="CIDEr",  # 监控CIDEr指标
            mode="max",          # 取最大值
            save_top_k=-1,        # 保存最好的3个检查点
            every_n_epochs=1,  # 每个epoch后保存
            save_on_train_epoch_end=False,  # 在验证阶段结束后保存
        )
        return [checkpoint_callback]
    
   

    def prompt_wrap(self, img_embeds, atts_img):
        prompt=f'Human: <Img><ImageHere></Img> {self.prompt} \nAssistant:'
        batch_size = img_embeds.shape[0]
        p_before, p_after = prompt.split('<ImageHere>')
        p_before_tokens = self.llama_tokenizer(
            p_before, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
        p_after_tokens = self.llama_tokenizer(
            p_after, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
        p_before_embeds = self.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1)
        p_after_embeds = self.embed_tokens(p_after_tokens.input_ids).expand(batch_size, -1, -1)
        wrapped_img_embeds = torch.cat([p_before_embeds, img_embeds, p_after_embeds], dim=1)
        wrapped_atts_img = atts_img[:, :1].expand(-1, wrapped_img_embeds.shape[1])
        return wrapped_img_embeds, wrapped_atts_img
    
    
    def encode_img_memory(self, images,id):
        # image_embeds_v= []
        # for image in images:
        device = images[0].device
           
        #直接获取相似报告的特征
        selected_report_memory = self.get_similar_reports(id).to(device)  # [B, K, 768]
        report_memory_token =  self.llama_proj(self.output_proj(selected_report_memory)) 
            
            
        
        # else:
        # Hopfield 计算
        # memory_output = self.R_HopfieldLayers(image_embeds)
        # report_memory_token = self.llama_proj(self.output_proj(memory_output))
            
        inputs_llama =  report_memory_token
        atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(device)

        return inputs_llama, atts_llama
    
    
    def encode_img_similar(self, images):
        # image_embeds = []
        image_memory_features=[]
        for i in range(len(images)):
            image_embeds_pool_list = []
            for image in images[i]:               
                device = image.device
                image_embed_pool = self.visual_encoder(image)['pooler_output'].unsqueeze(1).to(device)
                image_embed_pool = reduce(image_embed_pool, 'b l e -> b e','mean')
                image_embeds_pool_list.append(image_embed_pool)
                
            image_embeds_pool_t = torch.stack(image_embeds_pool_list).mean(0)    
            
            image_memory_features.append(image_embeds_pool_t)
        
        image_memory_features = torch.stack(image_memory_features).mean(0).unsqueeze(1)
        inputs_llama = self.llama_proj(image_memory_features)
        atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(image.device)

        return inputs_llama, atts_llama
    def prompt_wrap_similar(self, img_embeds, atts_img):
        prompt=f'Note: <Img><ImageHere></Img> with similar disease. '
        # prompt=f'Note: <Img><ImageHere></Img> without similar disease. '

        batch_size = img_embeds.shape[0]
        p_before, p_after = prompt.split('<ImageHere>')
        p_before_tokens = self.llama_tokenizer(
            p_before, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
        p_after_tokens = self.llama_tokenizer(
            p_after, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
        p_before_embeds = self.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1)
        p_after_embeds = self.embed_tokens(p_after_tokens.input_ids).expand(batch_size, -1, -1)
        # print(p_before_embeds.size(),img_embeds.size(),p_after_embeds.size())
        wrapped_img_embeds = torch.cat([p_before_embeds, img_embeds, p_after_embeds], dim=1)
        wrapped_atts_img = atts_img[:, :1].expand(-1, wrapped_img_embeds.shape[1])
        # print(wrapped_atts_img.size(),atts_img.size())
        return wrapped_img_embeds, wrapped_atts_img
    def self_refining_loss(self, ht, ev):
        """
        计算 self-refining loss
        ht: (b, embed_dim) - 聚合文本表征
        ev: (b, embed_dim) - 聚合图像表征
        return: loss_refine (scalar)
        """
        similarity = torch.sum(ht * ev, dim=-1)  # 计算相似度
        loss_refine = torch.exp(-similarity).mean()  # 负指数损失
        return loss_refine
    def compute_token_embeddings(self, logits, embedding_matrix):
        """
        计算基于 Gumbel-Softmax 的 soft token 表征
        logits: (b, seq_len, vocab_size)  
        embedding_matrix: (vocab_size, embed_dim)
        return: token_embeddings (b, seq_len, embed_dim)
        """
        gumbel_output = F.gumbel_softmax(logits, tau=self.tau, hard=False, dim=-1)  # (b, seq_len, vocab_size)
        token_embeddings = torch.matmul(gumbel_output, embedding_matrix)  # (b, seq_len, embed_dim)
        return token_embeddings

    def aggregate_text_representation(self, token_embeddings, attention_weights):
        """
        计算加权文本表征 ht
        token_embeddings: (b, seq_len, embed_dim)
        attention_weights: (b, seq_len)  
        return: ht (b, embed_dim)
        """
        ht =  torch.bmm(attention_weights, token_embeddings) # (b, embed_dim)
        ht = ht.mean(dim=1)  # (b, embed_dim)
        ht_proj = self.proj_text_similar(ht)  # (b, 1024)

        return ht_proj
    
    def encode_img(self, images, segmentation=None):
        image_embeds = []
        for image in images:
            device = image.device
            image_embed = self.visual_encoder(image)['last_hidden_state'].to(device)
            image_embeds.append(image_embed)
            
        image_embeds = torch.stack(image_embeds).mean(0)
        
        inputs_llama = self.llama_proj(image_embeds)
        atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(image.device)

        return inputs_llama, atts_llama,image_embeds
    
    def prompt_wrap_similar_v2_0513(self, img_embeds, atts_img, similar_report=None):
        """
        包装输入，使得 LLaMA 能够生成更符合医学报告标准的 FINDINGS 。
        
        1. 提供系统指令，明确模型角色和任务。
        2. 如果有 similar_report，提供额外示例，确保 FINDINGS 语言风格一致。
        """
        
        system_instruction = (
            "You are an expert chest radiologist. Your task is to generate the FINDINGS section of a chest X-ray report based on the provided medical image. Ensure that your response is concise, medically accurate, and adheres to standard radiology terminology.\n"
        )
        
        example_section = ""
        if similar_report:  # 如果有相似报告，加入示例
            example_section = "Here are some example FINDINGS sections. Learn their style and wording consistency:\n"
            for i, report in enumerate(similar_report):
                example_section += f"Example {i+1}: {report}\n"
        
        prompt = f"Human: <Img><ImageHere></Img> {system_instruction}{example_section} \nAssistant:"
        
        batch_size = img_embeds.shape[0]
        p_before, p_after = prompt.split('<ImageHere>')
        
        p_before_tokens = self.llama_tokenizer(p_before, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
        p_after_tokens = self.llama_tokenizer(p_after, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
        
        p_before_embeds = self.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1)
        p_after_embeds = self.embed_tokens(p_after_tokens.input_ids).expand(batch_size, -1, -1)
        
        wrapped_img_embeds = torch.cat([p_before_embeds, img_embeds, p_after_embeds], dim=1)
        wrapped_atts_img = atts_img[:, :1].expand(-1, wrapped_img_embeds.shape[1])
        
        return wrapped_img_embeds, wrapped_atts_img
    
    
    
    def prompt_wrap_report(self, img_embeds, atts_img):
        prompt=f'Human: <Img><ImageHere></Img> {self.prompt_report} \nAssistant:'
        batch_size = img_embeds.shape[0]
        p_before, p_after = prompt.split('<ImageHere>')
        p_before_tokens = self.llama_tokenizer(
            p_before, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
        p_after_tokens = self.llama_tokenizer(
            p_after, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
        p_before_embeds = self.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1)
        p_after_embeds = self.embed_tokens(p_after_tokens.input_ids).expand(batch_size, -1, -1)
        wrapped_img_embeds = torch.cat([p_before_embeds, img_embeds, p_after_embeds], dim=1)
        wrapped_atts_img = atts_img[:, :1].expand(-1, wrapped_img_embeds.shape[1])
        return wrapped_img_embeds, wrapped_atts_img
    def forward(self, samples):
        
        # clip_features = samples['clip_memory']
        image = samples["image"]
        id =samples['id']
        # 视觉编码器处理
        image_memory = samples['similar_images_path']
        image_memory = rearrange(image_memory,'b k i c h w -> i k b c h w')
        img_embeds_similar, atts_img_similar = self.encode_img_similar(image_memory)
        img_embeds_similar = self.layer_norm(img_embeds_similar)
        img_embeds_similar, atts_img_similar = self.prompt_wrap_similar(img_embeds_similar, atts_img_similar)
        
        img_embeds, atts_img,_= self.encode_img(image)
        image_features = img_embeds
        img_embeds = self.layer_norm(img_embeds)
        img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img)
        
        similar_report = samples['similar_report']

        similar_report_img_embeds, simialr_report_atts_img = self.prompt_wrap_similar_v2_0513(img_embeds, atts_img,similar_report)
        
        # selected_report_memory =samples['similar report feature'].to(device)  # [B, K, 768]
        # selected_report_memory = self.get_similar_reports(samples).to(device)  # [B, K, 768]
        # img_embeds_report =  self.llama_proj(self.output_proj_report(selected_report_memory)) 
        # atts_img_report =torch.ones(img_embeds_report.size()[:-1], dtype=torch.long).to(device) 
        id =samples['id']
        img_embeds_report, atts_img_report= self.encode_img_memory(image,id)
        img_embeds_report = self.layer_norm(img_embeds_report)
        img_embeds_report, atts_img_report = self.prompt_wrap_report(img_embeds_report, atts_img_report)
        
        
        img_embeds =torch.concat((img_embeds_similar,similar_report_img_embeds,img_embeds,img_embeds_report),dim=1)
        atts_img =torch.concat((atts_img_similar,simialr_report_atts_img,atts_img,atts_img_report),dim=1)


        self.llama_tokenizer.padding_side = "right"
        text = [t + self.end_sym for t in samples["input_text"]]

        to_regress_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.hparams.max_length,
            add_special_tokens=False
        ).to(image[0].device)

        targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == 0, -100
        )

        empty_targets = (
            torch.ones([atts_img.shape[0], atts_img.shape[1]+1],
                       dtype=torch.long).to(image[0].device).fill_(-100)  # plus one for bos
        )
        targets = torch.cat([empty_targets, targets], dim=1)

        batch_size = img_embeds.shape[0]
        bos = torch.ones([batch_size, 1],
                         dtype=to_regress_tokens.input_ids.dtype,
                         device=to_regress_tokens.input_ids.device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.embed_tokens(bos)
        atts_bos = atts_img[:, :1]

        to_regress_embeds = self.embed_tokens(to_regress_tokens.input_ids)
        inputs_embeds = torch.cat([bos_embeds, img_embeds, to_regress_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, atts_img, to_regress_tokens.attention_mask], dim=1)

        outputs = self.llama_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_attentions=True,
            return_dict=True,
            labels=targets,
        )
        
       
        

        loss = outputs.loss
        
        
        logits = outputs.logits  # 从 LLaMA 获取 logits
        embedding_matrix = self.llama_model.get_input_embeddings().weight  # 获取词嵌入矩阵

        # 1. 用 Gumbel-Softmax 计算 soft token 表征
        # gumbel_output = F.gumbel_softmax(logits, tau=1.0, hard=False, dim=-1)  # (b, seq_len, vocab_size)
        # text_embeddings = torch.matmul(gumbel_output, embedding_matrix)  # (b, seq_len, 1024)
        # 2. 提取最后一层注意力权重（假设 outputs 包含 attentions）
        attention_weights = outputs.attentions[-1].mean(dim=1)  # (b, seq_len)

        # 1. 计算 token 表征
        token_embeddings = self.compute_token_embeddings(logits, embedding_matrix)

        # 2. 计算文本表征 ht
        ht = self.aggregate_text_representation(token_embeddings, attention_weights)
        
        ev = self.attention_pooling_similar(image_features)  # (b, embed_dim)
        
        # 4. 计算 self-refining loss
        loss_refine = self.self_refining_loss(ht, ev)

        loss = self.lambda_report*outputs.loss +self.lambda_refine*loss_refine
        
       
        
        
       

 
        return {"loss": loss}
    
       

 
        # return {"loss": loss}
    
    

    def training_step(self, batch, batch_idx):
        result = self(batch)
        self.log_dict(result, prog_bar=True)
        return result
    
    
    def save_checkpoint(self, eval_res):
        current_epoch, global_step = self.trainer.current_epoch, self.trainer.global_step
        
        # 直接獲取完整的 state_dict，不進行過濾
        state_dict = self.state_dict()
        
        save_obj = {
            "model": state_dict,  # 包含所有參數
            "config": self.hparams,
            "epoch": current_epoch,
            "step": global_step,
            "optimizer": self.trainer.optimizers[0].state_dict(),  # 通常也會保存優化器狀態
            "eval_res": eval_res  # 保存評估結果
        }
        
        os.makedirs(os.path.join(self.hparams.savedmodel_path, 'checkpoints'), exist_ok=True)
        save_to = os.path.join(
            self.hparams.savedmodel_path, 
            'checkpoints',
            "checkpoint_epoch{}_step{}_bleu{:3f}_cider{:3f}.pth".format(
                current_epoch, 
                global_step, 
                eval_res['Bleu_4'], 
                eval_res['CIDEr']
            ),
        )
        self.print("Saving checkpoint at step {} to {}.".format(global_step, save_to))
        torch.save(save_obj, save_to)

    # def save_checkpoint(self, eval_res):
    #     current_epoch, global_step = self.trainer.current_epoch, self.trainer.global_step
    #     param_grad_dic = {
    #         k: v.requires_grad for (k, v) in self.named_parameters() if v.requires_grad
    #     }
    #     state_dict = self.state_dict()
    #     for k in list(state_dict.keys()):
    #         if k not in param_grad_dic.keys():
    #             del state_dict[k]
    #     save_obj = {
    #         "model": state_dict,
    #         "config": self.hparams,
    #         "epoch": current_epoch,
    #         "step":global_step
    #     }
    #     os.makedirs(os.path.join(self.hparams.savedmodel_path, 'checkpoints'), exist_ok=True)
    #     save_to = os.path.join(
    #         self.hparams.savedmodel_path, 'checkpoints',
    #         "checkpoint_epoch{}_step{}_bleu{:3f}_cider{:3f}.pth".format(current_epoch, global_step, eval_res['Bleu_4'], eval_res['CIDEr']),
    #     )
    #     self.print("Saving checkpoint at step {} to {}.".format(global_step, save_to))
    #     torch.save(save_obj, save_to)
    
    def validation_step(self, samples, batch_idx):
        self.llama_tokenizer.padding_side = "right"
        to_regress_tokens = self.llama_tokenizer(
            samples['input_text'],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.hparams.max_length,
            add_special_tokens=False
        )
        
        image = samples["image"]
        device = image[0].device
        image_memory = samples['similar_images_path'][0]
        image_memory = rearrange(image_memory,'b k i c h w -> i k b c h w')
        img_embeds_similar, atts_img_similar = self.encode_img_similar(image_memory)
        img_embeds_similar = self.layer_norm(img_embeds_similar)
        img_embeds_similar, atts_img_similar = self.prompt_wrap_similar(img_embeds_similar, atts_img_similar)
        
        img_embeds, atts_img,image_features= self.encode_img(image)
        img_embeds = self.layer_norm(img_embeds)
        img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img)
        
        similar_report = samples['similar_report_v2_0513'][:5]

        similar_report_img_embeds, simialr_report_atts_img = self.prompt_wrap_similar_v2_0513(img_embeds, atts_img,similar_report)
        # selected_report_memory =samples['similar report feature'].to(device)  # [B, K, 768]
        # selected_report_memory = self.get_similar_reports(samples).to(device)  # [B, K, 768]
        # img_embeds_report =  self.llama_proj(self.output_proj_report(selected_report_memory)) 
        # atts_img_report =torch.ones(img_embeds_report.size()[:-1], dtype=torch.long).to(device) 
        id =samples['id']
        img_embeds_report, atts_img_report= self.encode_img_memory(image,id)
        img_embeds_report = self.layer_norm(img_embeds_report)
        img_embeds_report, atts_img_report = self.prompt_wrap_report(img_embeds_report, atts_img_report)
        
        
        img_embeds =torch.concat((img_embeds_similar,similar_report_img_embeds,img_embeds,img_embeds_report),dim=1)
        atts_img =torch.concat((atts_img_similar,simialr_report_atts_img,atts_img,atts_img_report),dim=1)

        
        

        batch_size = img_embeds.shape[0]
        bos = torch.ones([batch_size, 1],
                         dtype=atts_img.dtype,
                         device=atts_img.device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.embed_tokens(bos)
        atts_bos = atts_img[:, :1]

        inputs_embeds = torch.cat([bos_embeds, img_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, atts_img], dim=1)

        outputs = self.llama_model.generate(
            inputs_embeds=inputs_embeds,
            num_beams=self.hparams.beam_size,
            do_sample=self.hparams.do_sample,
            min_new_tokens=self.hparams.min_new_tokens,
            max_new_tokens=self.hparams.max_new_tokens,
            repetition_penalty=self.hparams.repetition_penalty,
            length_penalty=self.hparams.length_penalty,
            temperature=self.hparams.temperature,
        )
        hypo = [self.decode(i) for i in outputs]
        ref = [self.decode(i) for i in to_regress_tokens['input_ids']]
        self.val_step_outputs.append({"hypo": hypo, "ref": ref, "id": samples["id"]})
        return hypo, ref
    
    def decode(self, output_token):
        if output_token[0] == 0:  # the model might output a unknow token <unk> at the beginning. remove it
            output_token = output_token[1:]
        if output_token[0] == 1:  # some users find that there is a start token <s> at the beginning. remove it
            output_token = output_token[1:]
        output_text = self.llama_tokenizer.decode(output_token, add_special_tokens=False)
        output_text = output_text.split('</s>')[0].strip()
        output_text = output_text.replace('<unk>', '')
        return output_text
    
    
    

    def on_validation_epoch_end(self):
        ref, hypo, ids = [], [], []
        for i in self.val_step_outputs:
            ref.extend(i['ref'])
            hypo.extend(i['hypo'])
            ids.extend(i['id'])

        ref = {k:[v] for k, v in zip(ids, ref)}
        hypo = {k:[v] for k, v in zip(ids, hypo)}
        eval_res = self.score(ref=ref,hypo=hypo)
        self.log_dict(eval_res, sync_dist=True, logger=True)

        result_folder = os.path.join(self.hparams.savedmodel_path, 'result')
        os.makedirs(result_folder, exist_ok=True)
        current_epoch, global_step = self.trainer.current_epoch, self.trainer.global_step
        json.dump(hypo, open(os.path.join(result_folder, f"result_{current_epoch}_{global_step}" + '.json'), 'w'))
        json.dump(ref, open(os.path.join(result_folder, 'refs.json'), 'w'))
        self.print(eval_res)

        val_score = 0
        for score_type, weight in zip(self.hparams.scorer_types, self.hparams.weights):
            val_score += eval_res[score_type] * weight

        if self.trainer.local_rank == 0:
            # if val_score > self.val_score:
            # self.save_checkpoint(eval_res)
            self.val_score = val_score
        self.val_step_outputs.clear()


    def test_step(self, samples, batch_idx):
        self.llama_tokenizer.padding_side = "right"
        to_regress_tokens = self.llama_tokenizer(
            samples['input_text'],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.hparams.max_length,
            add_special_tokens=False
        )

        image = samples["image"]
        image_memory = samples['similar_images_path']
        image_memory = rearrange(image_memory,'b k i c h w -> i k b c h w')
        img_embeds_similar, atts_img_similar = self.encode_img_similar(image_memory)
        img_embeds_similar = self.layer_norm(img_embeds_similar)
        img_embeds_similar, atts_img_similar = self.prompt_wrap_similar(img_embeds_similar, atts_img_similar)
        
        img_embeds, atts_img,image_features= self.encode_img(image)
        img_embeds = self.layer_norm(img_embeds)
        img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img)
        
        id =samples['id']
        img_embeds_report, atts_img_report= self.encode_img_memory(image,id)
        img_embeds_report = self.layer_norm(img_embeds_report)
        img_embeds_report, atts_img_report = self.prompt_wrap_report(img_embeds_report, atts_img_report)
        
        similar_report = samples['similar_repor']

        similar_report_img_embeds, simialr_report_atts_img = self.prompt_wrap_similar_v2_0513(img_embeds, atts_img,similar_report)
        
        
        img_embeds =torch.concat((img_embeds_similar,similar_report_img_embeds,img_embeds,img_embeds_report),dim=1)
        atts_img =torch.concat((atts_img_similar,simialr_report_atts_img,atts_img,atts_img_report),dim=1)
        

        
        

        batch_size = img_embeds.shape[0]
        bos = torch.ones([batch_size, 1],
                         dtype=atts_img.dtype,
                         device=atts_img.device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.embed_tokens(bos)
        atts_bos = atts_img[:, :1]

        inputs_embeds = torch.cat([bos_embeds, img_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, atts_img], dim=1)

        outputs = self.llama_model.generate(
            inputs_embeds=inputs_embeds,
            num_beams=self.hparams.beam_size,
            do_sample=self.hparams.do_sample,
            min_new_tokens=self.hparams.min_new_tokens,
            max_new_tokens=self.hparams.max_new_tokens,
            repetition_penalty=self.hparams.repetition_penalty,
            length_penalty=self.hparams.length_penalty,
            temperature=self.hparams.temperature,
        )
        hypo = [self.decode(i) for i in outputs]
        ref = [self.decode(i) for i in to_regress_tokens['input_ids']]
        self.test_step_outputs.append({"hypo": hypo, "ref": ref, "id": samples["id"]})
        return hypo, ref


    def on_test_epoch_end(self):
        """
        This function is called at the end of the test epoch.
        It is recommended to test on single device to ensure each sample/batch gets evaluated exactly once. This is helpful to make sure benchmarking for research papers is done the right way. Otherwise, in a multi-device setting, samples could occur duplicated when DistributedSampler is used, for eg. with strategy="ddp". It replicates some samples on some devices to make sure all devices have same batch size in case of uneven inputs.
        """
        ref, hypo, ids = [], [], []
        for i in self.test_step_outputs:
            ref.extend(i['ref'])
            hypo.extend(i['hypo'])
            ids.extend(i['id'])

        ref = {k:[v] for k, v in zip(ids, ref)}
        hypo = {k:[v] for k, v in zip(ids, hypo)}
        eval_res = self.score(ref=ref,hypo=hypo)

        result_folder = os.path.join(self.hparams.savedmodel_path, 'result')
        os.makedirs(result_folder, exist_ok=True)
        json.dump(hypo, open(os.path.join(result_folder, f"test_result.json"), 'w'))
        json.dump(ref, open(os.path.join(result_folder, 'test_refs.json'), 'w'))
        self.print(f"Test result of {self.hparams.delta_file}: {eval_res}")

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.learning_rate, momentum=0.9)
        # optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.hparams.max_epochs, eta_min=1e-6)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
        optimizer.zero_grad()
        
class ImageAttentionPooling_similar(nn.Module):
    """
    用 Attention 计算 image_repr
    """
    def __init__(self, embed_dim):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim))  # learnable query
        self.attention = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True)

    def forward(self, image_embeds):
        batch_size = image_embeds.shape[0]
        query = self.query.expand(batch_size, -1, -1)  # (b, 1, embed_dim)
        attn_output, _ = self.attention(query, image_embeds, image_embeds)
        image_repr = attn_output.squeeze(1)  # (b, embed_dim)
        return image_repr