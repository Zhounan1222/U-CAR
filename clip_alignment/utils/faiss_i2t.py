import faiss
import numpy as np
import json

# 读取clip对齐后存储特征
with open("image_features.json", "r") as f:
    image_features = json.load(f)

with open("train_report_features.json", "r") as f:
    train_report_features = json.load(f)

# 确保 image 和 train_report 是对齐的
image_ids = list(image_features.keys())  
train_report_ids = list(train_report_features.keys())  

# 构建 NumPy 矩阵
image_matrix = np.array([image_features[img_id] for img_id in image_ids]).astype("float32")
train_report_matrix = np.array([train_report_features[rep_id] for rep_id in train_report_ids]).astype("float32")

# FAISS 索引（余弦相似度 = 内积）
index = faiss.IndexFlatIP(train_report_matrix.shape[1])  
index.add(train_report_matrix)  

# 计算相似度
D, I = index.search(image_matrix, 20)  
image_report_similarity = {}

for i, image_id in enumerate(image_ids):
    similar_reports = []

    for j in range(len(I[i])):
        report_id = train_report_ids[I[i][j]]

        # **确保不是原本的报告**
        if image_id != report_id:
            # 获取该报告的特征
            report_feature = train_report_features[report_id]

            # 添加该报告特征到相似报告列表
            similar_reports.append(report_feature)

    # 取前 20 个最相似的报告特征（如果有足够多）
    image_report_similarity[image_id] = similar_reports[:20]


# 保存 JSON
with open("image_to_report_similarity_mimic.json", "w") as f:
    json.dump(image_report_similarity, f)



