import torch
from transformers import BertForSequenceClassification, BertTokenizer

model = BertForSequenceClassification.from_pretrained("LLM_Model_Game")
tokenizer = BertTokenizer.from_pretrained("LLM_Model_Game")

print(torch.cuda.is_available()) # True = using GPU
print(torch.cuda.device_count())

# ====== Test a prediction ======
#test_text = "I love DES329/331"
test_text = "I use Pikachu to win gym 2"
# Tokenize
# use softmax to see probability
inputs = tokenizer(test_text, return_tensors="pt", truncation=True, padding=True)
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    probs = torch.softmax(logits, dim=1)
    score = probs[0][1].item() # Probability for class 1 (sexual)
    score0 = probs[0][0].item() # Probability for class 1 (sexual)
    predicted_class = probs.argmax().item()


# Show result
print(f"🧠 Input: {test_text}")
print(f"🔎 Predicted class: {predicted_class} (1 = Game, 0 = not Game)")
print(f"📊 Score for class 1 (Game): {score:.4f}")
print(f"📊 Score for class 0 (No Game): {score0:.4f}")

