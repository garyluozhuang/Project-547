# import pandas as pd
# import numpy as np
# import faiss 
# from faiss import normalize_L2
# from tqdm import tqdm

# df = pd.read_csv('result_with_embedding.csv')
# df = df.reset_index()


# # # For building index
# # ids = []
# # datas = []
# # for i in df['index']:
# #     ids.append(i)
# #     x = df[df['index'] == i]['embedding'].values[0]
# #     datas.append(np.array([float(x) for x in x[1:-1].split(', ')]))
# # ids = np.asarray(ids).astype(np.int64)
# # datas = np.array(datas).astype(np.float32)
# # normalize_L2(datas)
# # quantizer = faiss.IndexFlatIP(datas.shape[1])
# # index = faiss.IndexIDMap(quantizer)
# # index.add_with_ids(datas.astype(np.float32), ids)
# # faiss.write_index(index, 'domain_content.index')

# # For making query
# index = faiss.read_index('domain_content.index')

# with open('astronomy_concepts.txt', 'r') as f:
#     astronomy_concepts = f.read().splitlines()
# astronomy_concepts = list(set(astronomy_concepts))

# from angle_emb import AnglE
# angle = AnglE.from_pretrained('WhereIsAI/UAE-Large-V1', pooling_strategy='cls').cuda()
# search_text = astronomy_concepts
# index_count_dict = {}
# for i in tqdm(range(len(search_text))):
#     vec = angle.encode([search_text[i]], to_numpy=True)
#     embedding_search = vec.astype(np.float32)
#     normalize_L2(embedding_search)
#     _, distance, idx = index.range_search(embedding_search.astype(np.float32), 0.58)
#     print(distance)
#     print(idx)
#     print('--------')
#     for item in idx:
#         if item not in index_count_dict:
#             index_count_dict[item] = 1
#         else:
#             index_count_dict[item] += 1
#     print('--------')

# sorted_dict = sorted(index_count_dict.items(), key=lambda item: item[1], reverse=True)
# print(sorted_dict)
# print(len(sorted_dict))
# result_idx = []
# for item in sorted_dict:
#     result_idx.append(item[0])
# df = df[df['index'].isin(result_idx)]
# df.to_csv('small_dataset_12k.csv', index=False)



# Prompt
# Can you give me 100 keywords or concepts in the area of astronomy and put them in a python list?
# Give me 50 common benchmark questions and the according answers in the area of astronomy, plz combine a question and an answer together as a str, and give me all the result as a python list.
# Give me a set of instruction to express the desire of wanting a astronomy-specific model to solve a astronomy-specific task, and put them in a python list. I will give you an example: I want the model to be an expert in astronomy and describe each planet well. Don't focus too much on the task of describing a planet, you can think of other astronomy-related tasks.


import pandas as pd
df = pd.read_csv('small_dataset_5k.csv')
for string in df['embedding_str']:
    if "corona" in string:
        print(string)