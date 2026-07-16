import fitz
import os
pdf_path = "data/attention-is-all-you-need.pdf"

doc = fitz.open(pdf_path)
os.makedirs("images", exist_ok=True)

all_text = ""

for page in doc:
    all_text += page.get_text()

image_count = 0

for page_num, page in enumerate(doc):

    images = page.get_images(full=True)

    for img_index, img in enumerate(images):

        xref = img[0]

        pix = fitz.Pixmap(doc, xref)

        if pix.n < 5:
            pix.save(f"images/page_{page_num+1}_{img_index+1}.png")
        else:
            pix = fitz.Pixmap(fitz.csRGB, pix)
            pix.save(f"images/page_{page_num+1}_{img_index+1}.png")

        image_count += 1

with open("paper_text.txt", "w", encoding="utf-8") as f:
    f.write(all_text)

print("Extraction Completed")
print("Characters:", len(all_text))
print("Images:", image_count)