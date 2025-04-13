import os
import shutil
import random
from pathlib import Path

def create_unbalanced_dataset():
    source_dir = Path("Aerial_Landscapes")
    target_dir = Path("Aerial_Landscapes_Unbalanced")

    if target_dir.exists():
        shutil.rmtree(target_dir)
    shutil.copytree(source_dir, target_dir)
    
    for class_dir in target_dir.iterdir():
        if not class_dir.is_dir():
            continue
        
        all_images = list(class_dir.glob("*.*"))
        total_origin = len(all_images)
        
        delete_num = random.randint(100, 700)
        keep_num = total_origin - delete_num
        
        if keep_num <= 0:
            keep_num = 1
        
        random.shuffle(all_images)
        keep_images = all_images[:keep_num]
        
        for img_path in all_images:
            if img_path not in keep_images:
                img_path.unlink()
        
        print(f"Created:{class_dir.name} | Original: {total_origin} | Kept: {keep_num}")

if __name__ == "__main__":
    create_unbalanced_dataset()