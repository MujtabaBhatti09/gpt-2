# from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
# import torch

# # Load the model and tokenizer
# model = GPT2LMHeadModel.from_pretrained('gpt2')
# tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# def tokenize_function(examples):
#     return tokenizer(examples, return_tensors='pt', max_length=512, truncation=True)

# # Example dataset (replace with actual tokenized data)
# data = ["This is some sample text to fine-tune the model."] * 100

# # Tokenizing the dataset
# inputs = tokenizer(data, return_tensors="pt", max_length=512, truncation=True)

# # Prepare dataset for training
# class CustomDataset(torch.utils.data.Dataset):
#     def __init__(self, encodings):
#         self.encodings = encodings

#     def __getitem__(self, idx):
#         return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

#     def __len__(self):
#         return len(self.encodings.input_ids)

# train_dataset = CustomDataset(inputs)

# # Fine-tuning settings
# training_args = TrainingArguments(
#     output_dir='./results',       # output directory
#     per_device_train_batch_size=2,  # batch size per device
#     num_train_epochs=3,            # number of training epochs
#     save_steps=10_000,
#     save_total_limit=2,
# )

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
# )

# # Fine-tune the model
# trainer.train()

# # Save the fine-tuned model
# model.save_pretrained('./fine_tuned_model')


# import torch
# from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments

# # Load GPT-2 model and tokenizer
# model = GPT2LMHeadModel.from_pretrained('gpt2')
# tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# # Tokenize your dataset
# def tokenize_data(data, max_length=512):
#     return tokenizer(data, return_tensors="pt", max_length=max_length, truncation=True, padding="max_length")

# # Example data - Replace this with your actual dataset
# data = [
#     "The quick brown fox jumps over the lazy dog.",
#     "GPT-2 is a powerful language model developed by OpenAI.",
#     "Fine-tuning language models helps improve task-specific performance."
# ]

# # Tokenize the data
# encodings = tokenizer(data, truncation=True, padding=True, max_length=512)

# # Custom dataset class for GPT-2
# class CustomDataset(torch.utils.data.Dataset):
#     def __init__(self, encodings):
#         self.encodings = encodings

#     def __getitem__(self, idx):
#         # Return input_ids, attention_mask, and labels (which are the same as input_ids)
#         return {
#             'input_ids': torch.tensor(self.encodings['input_ids'][idx]),
#             'attention_mask': torch.tensor(self.encodings['attention_mask'][idx]),
#             'labels': torch.tensor(self.encodings['input_ids'][idx])  # GPT-2 uses input_ids as labels
#         }

#     def __len__(self):
#         return len(self.encodings['input_ids'])

# # Create a dataset from the tokenized data
# dataset = CustomDataset(encodings)

# # Define training arguments
# training_args = TrainingArguments(
#     output_dir='./results',           # Output directory for model checkpoints
#     num_train_epochs=3,               # Number of training epochs
#     per_device_train_batch_size=2,    # Batch size per device
#     save_steps=10_000,                # Save checkpoint every 10,000 steps
#     save_total_limit=2,               # Limit the total number of checkpoints
#     logging_dir='./logs',             # Directory for storing logs
#     logging_steps=200,                # Log every 200 steps
#     evaluation_strategy="no",         # Disable evaluation
# )

# # Initialize the Trainer
# trainer = Trainer(
#     model=model,                        # The model to train
#     args=training_args,                 # Training arguments
#     train_dataset=dataset               # Training dataset
# )

# # Fine-tune the GPT-2 model
# trainer.train()

# # Save the fine-tuned model
# model.save_pretrained('./fine_tuned_model')
# tokenizer.save_pretrained('./fine_tuned_model')

# print("Fine-tuning completed and model saved!")

# import torch
# from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments

# # Load GPT-2 model and tokenizer
# model = GPT2LMHeadModel.from_pretrained('gpt2')
# tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# # Fix for padding token issue
# tokenizer.pad_token = tokenizer.eos_token  # Alternatively, use tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# # Tokenize your dataset
# def tokenize_data(data, max_length=512):
#     return tokenizer(data, return_tensors="pt", max_length=max_length, truncation=True, padding="max_length")

# # Example data - Replace this with your actual dataset
# data = [
#     "The quick brown fox jumps over the lazy dog.",
#     "GPT-2 is a powerful language model developed by OpenAI.",
#     "Fine-tuning language models helps improve task-specific performance."
# ]

# # Tokenize the data
# encodings = tokenizer(data, truncation=True, padding=True, max_length=512)

# # Custom dataset class for GPT-2
# class CustomDataset(torch.utils.data.Dataset):
#     def __init__(self, encodings):
#         self.encodings = encodings

#     def __getitem__(self, idx):
#         # Return input_ids, attention_mask, and labels (which are the same as input_ids)
#         return {
#             'input_ids': torch.tensor(self.encodings['input_ids'][idx]),
#             'attention_mask': torch.tensor(self.encodings['attention_mask'][idx]),
#             'labels': torch.tensor(self.encodings['input_ids'][idx])  # GPT-2 uses input_ids as labels
#         }

#     def __len__(self):
#         return len(self.encodings['input_ids'])

# # Create a dataset from the tokenized data
# dataset = CustomDataset(encodings)

# # Define training arguments
# training_args = TrainingArguments(
#     output_dir='./results',           
#     num_train_epochs=5,               # Increase the number of epochs
#     per_device_train_batch_size=2,    
#     save_steps=10_000,                
#     save_total_limit=2,               
#     logging_dir='./logs',             
#     logging_steps=200,                
# )
# # training_args = TrainingArguments(
# #     output_dir='./results',           # Output directory for model checkpoints
# #     num_train_epochs=3,               # Number of training epochs
# #     per_device_train_batch_size=2,    # Batch size per device
# #     save_steps=10_000,                # Save checkpoint every 10,000 steps
# #     save_total_limit=2,               # Limit the total number of checkpoints
# #     logging_dir='./logs',             # Directory for storing logs
# #     logging_steps=200,                # Log every 200 steps
# #     evaluation_strategy="no",         # Disable evaluation
# # )

# # Initialize the Trainer
# trainer = Trainer(
#     model=model,                        # The model to train
#     args=training_args,                 # Training arguments
#     train_dataset=dataset               # Training dataset
# )

# # Fine-tune the GPT-2 model
# trainer.train()

# # Save the fine-tuned model
# model.save_pretrained('./fine_tuned_model')
# tokenizer.save_pretrained('./fine_tuned_model')

# print("Fine-tuning completed and model saved!")

# import torch
# from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments

# # Load GPT-2 model and tokenizer
# model = GPT2LMHeadModel.from_pretrained('gpt2')
# tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# # Load the filtered dataset
# with open("../data/filtered_dataset.txt", "r", encoding="utf-8") as f:
#     data = f.readlines()

# # Tokenize the dataset
# encodings = tokenizer(data, truncation=True, padding=True, max_length=512, return_tensors='pt')

# # Custom dataset class
# class CustomDataset(torch.utils.data.Dataset):
#     def __init__(self, encodings):
#         self.encodings = encodings

#     def __getitem__(self, idx):
#         return {
#             'input_ids': self.encodings['input_ids'][idx],
#             'attention_mask': self.encodings['attention_mask'][idx],
#             'labels': self.encodings['input_ids'][idx]
#         }

#     def __len__(self):
#         return len(self.encodings['input_ids'])

# # Create the dataset
# dataset = CustomDataset(encodings)

# # Define training arguments
# training_args = TrainingArguments(
#     output_dir='./results',
#     num_train_epochs=3,                # You can increase this for more training time
#     per_device_train_batch_size=2,
#     save_steps=10_000,
#     save_total_limit=2,
#     logging_dir='./logs',
#     logging_steps=200,
# )

# # Initialize the Trainer
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=dataset
# )

# # Fine-tune the model
# trainer.train()

# # Save the fine-tuned model
# model.save_pretrained('./fine_tuned_model')
# tokenizer.save_pretrained('./fine_tuned_model')

# print("Fine-tuning completed and model saved!")

# import torch
# from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments

# # Set absolute paths for data and model saving
# DATA_FILE = r'D:\Project\FYP\pdf-bot\data\filtered_dataset.txt'
# OUTPUT_DIR = r'D:\Project\FYP\pdf-bot\fine_tuned_model'

# # Load GPT-2 model and tokenizer
# model = GPT2LMHeadModel.from_pretrained('gpt2')
# tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# # Add a pad token if it's missing
# if tokenizer.pad_token is None:
#     tokenizer.pad_token = tokenizer.eos_token  # Alternatively, use tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# # Load the filtered dataset
# with open(DATA_FILE, "r", encoding="utf-8") as f:
#     data = f.readlines()

# # Tokenize the dataset
# encodings = tokenizer(data, truncation=True, padding=True, max_length=512, return_tensors='pt')

# # Custom dataset class
# class CustomDataset(torch.utils.data.Dataset):
#     def __init__(self, encodings):
#         self.encodings = encodings

#     def __getitem__(self, idx):
#         return {
#             'input_ids': self.encodings['input_ids'][idx],
#             'attention_mask': self.encodings['attention_mask'][idx],
#             'labels': self.encodings['input_ids'][idx]
#         }

#     def __len__(self):
#         return len(self.encodings['input_ids'])

# # Create the dataset
# dataset = CustomDataset(encodings)

# # # Define training arguments
# # training_args = TrainingArguments(
# #     output_dir=OUTPUT_DIR,  # Save model in the fine_tuned_model directory
# #     num_train_epochs=3,     # You can increase this for more training time
# #     per_device_train_batch_size=2,
# #     save_steps=10_000,
# #     save_total_limit=2,
# #     logging_dir='./logs',
# #     logging_steps=200,
# # )

# training_args = TrainingArguments(
#     output_dir=OUTPUT_DIR,
#     num_train_epochs=3,
#     per_device_train_batch_size=4,  # Increase batch size
#     save_steps=50_000,               # Save less frequently
#     logging_steps=1_000,             # Log less frequently
#     evaluation_strategy="epoch",      # Evaluate every epoch
#     logging_dir='./logs',
#     fp16=True,                       # Enable mixed precision if using compatible hardware
# )

# # Initialize the Trainer
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=dataset
# )

# # Fine-tune the model
# trainer.train()

# # Save the fine-tuned model
# model.save_pretrained(OUTPUT_DIR)
# tokenizer.save_pretrained(OUTPUT_DIR)

# print("Fine-tuning completed and model saved!")

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments

# Set absolute paths for data and model saving
DATA_FILE = r'D:\Project\FYP\pdf-bot\data\all_structured_data.json'
OUTPUT_DIR = r'D:\Project\FYP\pdf-bot\fine_tuned_model'

# Load GPT-2 model and tokenizer
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Add a pad token if it's missing
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load the filtered dataset
with open(DATA_FILE, "r", encoding="utf-8") as f:
    data = f.readlines()

# Tokenize the dataset
encodings = tokenizer(data, truncation=True, padding=True, max_length=512, return_tensors='pt')

# Custom dataset class
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels': self.encodings['input_ids'][idx]
        }

    def __len__(self):
        return len(self.encodings['input_ids'])

# Create the dataset
dataset = CustomDataset(encodings)

# Define training arguments without evaluation
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=4,  # Increase batch size
    save_steps=50_000,               # Save less frequently
    logging_steps=1_000,             # Log less frequently
    evaluation_strategy="no",        # Disable evaluation
    logging_dir='./logs',
    fp16=True,                       # Enable mixed precision if using compatible hardware
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("Fine-tuning completed and model saved!")