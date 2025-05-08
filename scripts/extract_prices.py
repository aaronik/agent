import requests
from bs4 import BeautifulSoup
import re

url = "https://invertedstone.com/calculators/openai-pricing"

response = requests.get(url)
response.raise_for_status()
soup = BeautifulSoup(response.text, "html.parser")

# Extract all text and split by lines
text_lines = soup.get_text(separator='\n').split('\n')

pricing = {}

model_name = None
input_price = None
output_price = None

for line in text_lines:
    line = line.strip()
    # Match model name e.g. OpenAI - o3
    model_match = re.match(r'OpenAI - (.+)', line)
    if model_match:
        if model_name and input_price and output_price:
            # Save previous model data
            pricing[model_name] = {"input": input_price, "output": output_price}
        model_name = model_match.group(1)
        input_price = None
        output_price = None
        continue

    # Match input price line e.g. Input: $
    if line.startswith("Input:"):
        # input price should appear in next line
        continue

    # Match price line e.g. 10.00
    price_match = re.match(r'\$?(\d+(?:\.\d+)?)', line)
    if price_match:
        price_val = float(price_match.group(1))
        if input_price is None:
            input_price = price_val
        elif output_price is None:
            output_price = price_val

# Save last model
if model_name and input_price and output_price:
    pricing[model_name] = {"input": input_price, "output": output_price}

print(pricing)
