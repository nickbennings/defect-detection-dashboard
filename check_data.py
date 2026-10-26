from pathlib import Path
from PIL import Image

dataset = Path("data/bottle")

for split in ["train", "test"]:
    split_path = dataset / split
    print(f"\n{split.upper()} images:")

    for category in sorted(split_path.iterdir()):
        if not category.is_dir():
            continue

        images = list(category.glob("*.png"))
        errors = 0

        for image_path in images:
            try:
                with Image.open(image_path) as image:
                    image.verify()
            except Exception as error:
                errors += 1
                print(f"  Could not open {image_path}: {error}")

        print(f"  {category.name}: {len(images)} images, {errors} errors")
        