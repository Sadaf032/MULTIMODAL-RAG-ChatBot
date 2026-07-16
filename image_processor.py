from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import os


# Load image captioning model
processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)


# Images folder
image_folder = "images"

descriptions = []


# Process images
for image_name in os.listdir(image_folder):

    if image_name.lower().endswith((".png", ".jpg", ".jpeg")):

        image_path = os.path.join(image_folder, image_name)

        image = Image.open(image_path).convert("RGB")


        # Generate caption
        inputs = processor(
            images=image,
            return_tensors="pt"
        )

        output = model.generate(**inputs)

        caption = processor.decode(
            output[0],
            skip_special_tokens=True
        )


        descriptions.append({
            "image": image_name,
            "description": caption
        })


        print(f"{image_name} -> {caption}")



# Save descriptions
with open(
    "image_descriptions.txt",
    "w",
    encoding="utf-8"
) as f:

    for item in descriptions:

        f.write(f"Image: {item['image']}\n")
        f.write(f"Description: {item['description']}\n")
        f.write("=" * 80)
        f.write("\n")


print("\nImage descriptions saved successfully!")