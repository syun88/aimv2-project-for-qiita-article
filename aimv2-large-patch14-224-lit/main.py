import requests
from PIL import Image
from transformers import AutoProcessor, AutoModel

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
text = ["Picture of a dog.", "Picture of a cat.", "Picture of a horse."]

processor = AutoProcessor.from_pretrained(
    "apple/aimv2-large-patch14-224-lit",
)
model = AutoModel.from_pretrained(
    "apple/aimv2-large-patch14-224-lit",
    trust_remote_code=True,
)

inputs = processor(
    images=image,
    text=text,
    add_special_tokens=True,
    truncation=True,
    padding=True,
    return_tensors="pt",
)
outputs = model(**inputs)
probs = outputs.logits_per_image.softmax(dim=-1)
print(outputs)