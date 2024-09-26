# from transformers import GPT2LMHeadModel, GPT2Tokenizer

# # Load the fine-tuned model and tokenizer from the same directory
# model = GPT2LMHeadModel.from_pretrained('./fine_tuned_model')
# tokenizer = GPT2Tokenizer.from_pretrained('./fine_tuned_model')

# # Add a pad token if it's missing
# if tokenizer.pad_token is None:
#     tokenizer.add_special_tokens({'pad_token': '[PAD]'})  # Set a distinct pad token

# # Resize model embeddings to account for newly added tokens (like the [PAD] token)
# model.resize_token_embeddings(len(tokenizer))

# def generate_response(prompt):
#     """Generates a response for the given prompt."""
#     # Tokenize the input and create attention masks
#     inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True, max_length=512)
    
#     # Generate text with proper attention mask and decoding settings
#     outputs = model.generate(
#         inputs['input_ids'], 
#         attention_mask=inputs['attention_mask'],   # Explicitly pass the attention mask
#         max_length=100, 
#         num_return_sequences=1,
#         no_repeat_ngram_size=2, 
#         do_sample=True, 
#         top_k=50,                                  # Limits the top 50 tokens for sampling
#         top_p=0.95,                                # Cumulative probability for nucleus sampling
#         temperature=0.7,                           # Controls randomness in generation
#         pad_token_id=tokenizer.pad_token_id        # Ensure the pad token is handled correctly
#     )
    
#     # Decode the generated text
#     response = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     return response

# # Example usage
# if __name__ == '__main__':
#     while True:
#         user_input = input("Enter Prompt: ")
        
#         if user_input.lower() == "exit":
#             print("Exiting The Chat. Goodbye!")
#             break
        
#         # Generate response
#         response = generate_response(user_input)
#         print("\nResponse:", response)


# from transformers import GPT2LMHeadModel, GPT2Tokenizer

# # Set absolute paths to the fine-tuned model and tokenizer
# MODEL_DIR = r'D:\Project\FYP\pdf-bot\fine_tuned_model'

# # Load the fine-tuned model and tokenizer
# model = GPT2LMHeadModel.from_pretrained(MODEL_DIR)
# tokenizer = GPT2Tokenizer.from_pretrained(MODEL_DIR)

# def generate_response(prompt):
#     """Generates a response for the given prompt."""
#     inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True, max_length=512)
#     outputs = model.generate(inputs['input_ids'], max_length=100, do_sample=True, top_p=0.95, top_k=50, temperature=0.7)
#     response = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     return response

# # Example usage
# if __name__ == '__main__':
#     while True:
#         user_input = input("Enter Prompt: ")
#         if user_input.lower() == "exit":
#             print("Exiting The Chat. Goodbye!")
#             break
#         response = generate_response(user_input)
#         print("Response:", response)

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the fine-tuned model and tokenizer
model = GPT2LMHeadModel.from_pretrained('D:\\Project\\FYP\\pdf-bot\\fine_tuned_model')
tokenizer = GPT2Tokenizer.from_pretrained('D:\\Project\\FYP\\pdf-bot\\fine_tuned_model')

# Set pad_token_id to eos_token_id to avoid issues
model.config.pad_token_id = model.config.eos_token_id

# Set the model to evaluation mode
model.eval()

while True:
    # Get user input
    prompt = input("Enter Prompt: ")
    if prompt == "quit":
        break

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors='pt')

    # Generate response, explicitly passing the attention mask
    outputs = model.generate(
        inputs['input_ids'],
        attention_mask=inputs['attention_mask'],  # Explicitly pass attention mask
        max_length=150,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id
    )

    # Decode and print the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Response: {response}")