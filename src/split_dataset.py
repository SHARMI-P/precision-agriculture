# src/split_dataset.py

import os, shutil, random

def split_dataset(
    source_dir = 'data/plantvillage/data',
    output_dir = 'data/plantvillage_split',
    val_split  = 0.2,
    seed       = 42
):
    random.seed(seed)

    class_folders = [
        f for f in os.listdir(source_dir)
        if os.path.isdir(os.path.join(source_dir, f))
    ]

    print(f"Found {len(class_folders)} classes")

    for cls in class_folders:
        cls_path = os.path.join(source_dir, cls)
        images   = [
            f for f in os.listdir(cls_path)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        random.shuffle(images)

        split_idx  = int(len(images) * (1 - val_split))
        train_imgs = images[:split_idx]
        val_imgs   = images[split_idx:]

        # Create output folders
        train_cls_dir = os.path.join(output_dir, 'train', cls)
        val_cls_dir   = os.path.join(output_dir, 'val',   cls)
        os.makedirs(train_cls_dir, exist_ok=True)
        os.makedirs(val_cls_dir,   exist_ok=True)

        # Copy images
        for img in train_imgs:
            shutil.copy(
                os.path.join(cls_path, img),
                os.path.join(train_cls_dir, img)
            )
        for img in val_imgs:
            shutil.copy(
                os.path.join(cls_path, img),
                os.path.join(val_cls_dir, img)
            )

        print(f"  {cls}: {len(train_imgs)} train | {len(val_imgs)} val")

    print(f"\nDone! Split dataset saved to: {output_dir}")

if __name__ == '__main__':
    split_dataset()