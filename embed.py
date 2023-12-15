import argparse
from datasets import load_dataset
from angle_emb import AnglE
import os
import numpy 
from tqdm import tqdm

def format_text(row):
    return f'USER:\n{row["question"]}\n\nASSISTANT:\n{row["choices"][int(row["answer"])]}'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=1024)
    args = parser.parse_args()
    
    embeddings = []
    mmlu_astronomy =  load_dataset("cais/mmlu", "astronomy")["auxiliary_train"]
    if args.rank == 0:
        data = mmlu_astronomy.select(range(0, len(mmlu_astronomy)//2))
    else:
        data = mmlu_astronomy.select(range(len(mmlu_astronomy)//2, len(mmlu_astronomy)))
    
    angle = AnglE.from_pretrained('WhereIsAI/UAE-Large-V1', pooling_strategy='cls').cuda(f"cuda:{args.rank}")
    data = list(data)
    for i in tqdm(range(0, len(data), args.batch_size)):
        batch = data[i:i+args.batch_size]
        text = [format_text(example) for example in batch]
        vec = angle.encode(text, to_numpy=True)
        embeddings.append(vec)
    
    embeddings = numpy.concatenate(embeddings, axis=0)
    numpy.save(f'embeddings_{args.rank}.npy', embeddings)
    print(f'Embeddings saved to embeddings_{args.rank}.npy')
    
        
        