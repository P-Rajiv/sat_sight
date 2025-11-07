import os, faiss, json, pickle, numpy as np
from tqdm import tqdm
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

META_PATH = 'data/image_metadata.jsonl'
IMG_ROOT = 'data'
INDEX_OUT = 'data/image_index.faiss'
VECS_OUT  = 'data/image_vectors.npy'
META_OUT = 'data/image_meta.pkl'


def load_meta():
    # Each line in META_PATH should contain a JSON object
    with open(META_PATH, 'r') as f:
        return [json.loads(line) for line in f]


def main():
    # ✅ Load CLIP model + processor
    model = CLIPModel.from_pretrained('laion/CLIP-ViT-H-14-laion2B-s32B-b79K')
    processor = CLIPProcessor.from_pretrained('laion/CLIP-ViT-H-14-laion2B-s32B-b79K')

    metas = load_meta()
    vecs, keep = [], []

    for m in tqdm(metas, desc="Embedding images"):
        path = os.path.join(IMG_ROOT, m['filename'])
        if not os.path.exists(path):
            continue

        img = Image.open(path).convert('RGB')

        # ✅ Preprocess and encode image
        inputs = processor(images=img, return_tensors="pt")
        with torch.no_grad():
            emb = model.get_image_features(**inputs)

        # ✅ Normalize embedding (for cosine similarity)
        emb = emb / emb.norm(p=2, dim=-1, keepdim=True)

        vecs.append(emb.cpu().numpy())
        keep.append(m)

    # ✅ Stack all embeddings into one array
    arr = np.vstack(vecs).astype('float32')

    # ✅ Build FAISS index (inner product = cosine similarity after normalization)
    index = faiss.IndexFlatIP(arr.shape[1])
    index.add(arr)

    # ✅ Save index and metadata
    faiss.write_index(index, INDEX_OUT)
    np.save(VECS_OUT, arr)

    with open(META_OUT, 'wb') as f:
        pickle.dump(keep, f)

    print(f'Successfully stored metadata in {META_OUT}')
    print(f'FAISS index saved at {INDEX_OUT}')
    print(f'Embeddings saved at {VECS_OUT}')


if __name__ == '__main__':
    import torch
    main()
