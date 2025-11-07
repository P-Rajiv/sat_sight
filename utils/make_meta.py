import os, json, random, shutil, argparse
from glob import glob

# Map EuroSAT classes -> plausible region + lat/lon bbox (synthetic!)
CLASS2REGION = {
    "Forest":            ("Amazon Basin, Brazil",      (-5.0, -1.0), (-66.0, -58.0), ["forest","tropical","river"]),
    "SeaLake":           ("Bay of Bengal",             (10.0, 20.0),  (85.0, 95.0),  ["coastal","water","sediment"]),
    "River":             ("Ganges Delta, India",       (21.0, 23.5),  (88.0, 91.0),  ["delta","flood","river"]),
    "Residential":       ("NCR, India",                (28.4, 28.9),  (76.8, 77.5),  ["urban","sprawl","roads"]),
    "Industrial":        ("Guangdong, China",          (22.5, 23.5),  (112.5, 114.5),["industrial","factories","urban"]),
    "Highway":           ("California, USA",           (33.0, 36.0),  (-119.0,-116.0),["highway","roads","dry"]),
    "AnnualCrop":        ("Punjab, India",             (30.5, 31.5),  (74.5, 76.0),  ["agriculture","irrigation","crop"]),
    "PermanentCrop":     ("Andalusia, Spain",          (36.5, 38.5),  (-6.5, -3.0),  ["olive","orchard","terrace"]),
    "HerbaceousVegetation":("Serengeti, Tanzania",     (-3.0,  -1.0), (34.0,  36.0), ["savanna","grassland","plain"]),
    "Pasture":           ("New South Wales, Australia",( -34.5,-32.0),(147.0,149.0), ["pasture","rangeland","fields"]),
    "BareSoil":          ("Sahara, Algeria",           (23.0, 26.0),  (3.0,  9.0),   ["desert","dune","arid"])
}

def pick_coords(lat_rng, lon_rng):
    return (random.uniform(*lat_rng), random.uniform(*lon_rng))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eurosat_root", required=True, help="Path to EuroSAT/2750")
    ap.add_argument("--out_images", default="data/images", help="Copy subset here")
    ap.add_argument("--out_meta", default="data/image_metadata.jsonl")
    ap.add_argument("--per_class", type=int, default=2, help="Images per class")
    args = ap.parse_args()

    os.makedirs(args.out_images, exist_ok=True)
    classes = [c for c in os.listdir(args.eurosat_root) if os.path.isdir(os.path.join(args.eurosat_root,c))]
    img_id = 1
    with open(args.out_meta, "w") as fout:
        for cls in classes:
            cls_dir = os.path.join(args.eurosat_root, cls)
            paths = sorted(glob(os.path.join(cls_dir, "*.jpg")))
            if not paths: 
                continue
            sample = random.sample(paths, min(args.per_class, len(paths)))
            region, lat_rng, lon_rng, tags = CLASS2REGION.get(cls, ("Unknown", (0,0), (0,0), [cls.lower()]))
            for p in sample:
                fn = f"{cls.lower()}_{img_id}.jpg"
                dst = os.path.join(args.out_images, fn)
                shutil.copy2(p, dst)
                lat, lon = pick_coords(lat_rng, lon_rng) if lat_rng != (0,0) else (None, None)
                meta = {
                    "image_id": f"IMG_{img_id:04d}",
                    "filename": f"images/{fn}",
                    "caption": f"{cls} scene (EuroSAT).",
                    "biome": cls,
                    "region_hint": region,
                    "lat": lat, "lon": lon,
                    "tags": tags,
                    "coords_synthetic": True,
                    "source": "EuroSAT RGB"
                }
                fout.write(json.dumps(meta) + "\n")
                img_id += 1
    print(f"Done. Wrote metadata to {args.out_meta}")

if __name__ == "__main__":
    main()
