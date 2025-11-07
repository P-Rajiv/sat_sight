import pickle, faiss, numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

IMG_PATH = "data/images"  # path to your images
INDEX_PATH = "data/image_index.faiss"
META_PATH = "data/image_meta.pkl"

def test_image(query_path):
    model = CLIPModel.from_pretrained('laion/CLIP-ViT-H-14-laion2B-s32B-b79K')
    processor = CLIPProcessor.from_pretrained('laion/CLIP-ViT-H-14-laion2B-s32B-b79K')
    index = faiss.read_index(INDEX_PATH)
    meta = pickle.load(open(META_PATH,"rb"))

    img = Image.open(query_path).convert("RGB")
    inputs = processor(images=img, return_tensors="pt")
    with torch.no_grad():
        q = model.get_image_features(**inputs)
        q = q / q.norm(p=2, dim=-1, keepdim=True)
    q = q.cpu().numpy().astype("float32")

    D,I = index.search(q, k=8)
    print("\nTop-8 visually similar images:")
    for rank,(score,idx) in enumerate(zip(D[0], I[0]),1):
        m = meta[idx]
        print(f"{rank}. {m['biome']} | {m['region_hint']} | cos={score:.3f} | tags={m['tags']}")

if __name__ == "__main__":
    import torch, glob
    imgs = sorted(glob.glob("data/images/*.jpg"))
    print("Query Image: ",imgs[0])
    test_image(imgs[0])
