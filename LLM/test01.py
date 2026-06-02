import transformers
from datasets import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer

print(transformers.__version__)
# Sample data
# 0 is good web
# 1 is pornography
#2 is gamble
#3 is game-content
#4 is hate-speech content

data = {
"text": [
"Hello, how are you today?",
"I will enroll in DES400 next semester",
"Tom yam Kung is a Thai signature dish",
"I see students playing Mario around SIIT rangsit campus",
"Good morning!",
"Cat is sleeping",
"HTTP stands for Hypertext protocol",
"My flight number is XJ601",
"Cordyceps is the fungi to control the deadly insects",
"It is almost impossible that humans will turn into a zombie in real world",
"Cordyceps can turn humans into aggressive zombie-like creatures",
"T-virus is rapidly widespread around the Raccon City and causes human turning into zombie-like creatures",
"You can play Ovenbreak on IOS28",
"RE Requiem is also ported to Nintendo Switch 2",
"Pikachu can evolve to Raichu by a thunder stone ",
"The DLC of separate way will be available after completing the main story",
"Leon is an agent in RE4",
"Leon said to Grace that I am just my eyes",
"The overall rating of CR7 in FC2026 is 90",
"You need to complete all eight gyms before the pokemon league"
],
"label": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
}

dataset = Dataset.from_dict(data)

# Load tokenizer and model
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Tokenize
def preprocess(examples):
    return tokenizer(examples["text"], truncation=True, padding=True)
tokenized = dataset.map(preprocess, batched=True)

# Split the dataset into train and validation sets (e.g., 80/20)
split_dataset = tokenized.train_test_split(test_size=0.2, seed=42)

print("*********************")
print(split_dataset)
print("*********************")

train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]


args = TrainingArguments(
output_dir="./results",
per_device_train_batch_size=8, # If GPU allows
num_train_epochs=4, # Try 3–5
evaluation_strategy="epoch", # Helpful to monitor
learning_rate=2e-5, # Stable for BERT
logging_dir="./logs",
save_strategy="epoch", # Save model at end of each epoch
save_total_limit=2
)


trainer = Trainer(
model=model,
args=args,
train_dataset=train_dataset,
eval_dataset=eval_dataset
)

trainer = Trainer(model=model,args=args,train_dataset=tokenized)

trainer.train()

# ====== Save fine-tuned model ======
model.save_pretrained("LLM_Model_Game")
tokenizer.save_pretrained("LLM_Model_Game")
print("✅ Model saved to 'LLM_Model_Game' folder")

