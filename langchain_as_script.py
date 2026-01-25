# %%
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

# %%
model_name = "deepset/roberta-base-squad2"  # Μοντέλο εκπαιδευμένο για QA
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# %%
# question = "Πόσο είναι το πρόστιμο;"
# context = "Σε περίπτωση καθυστέρησης, ο πελάτης επιβαρύνεται με πρόστιμο 500 ευρώ άμεσα."

question = "How much is the fine?"
context = "In case of a delay, the customer needs to pay 500 euro immediately."

# %%
inputs = tokenizer(question, context, return_tensors="pt")
print("Inputs:", inputs)

# %%
print("Tokens: ")
print(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]))

# %%
with torch.no_grad():
    outputs = model(**inputs)
    start_scores = outputs.start_logits
    end_scores = outputs.end_logits

print("Start scores:", start_scores)
print("End scores:", end_scores)

print("start_scores shape:", start_scores.shape)
print("end_scores shape:", end_scores.shape)


# %%
answer_start_index = torch.argmax(start_scores)
answer_end_index = torch.argmax(end_scores) + 1

print(f"Start Index: {answer_start_index}, End Index: {answer_end_index}")

# %%
if answer_end_index < answer_start_index:
    print("Error: Το μοντέλο δεν βρήκε έγκυρη απάντηση.")
else:
    # 3. Extraction (Slicing στα Input IDs)
    # Παίρνουμε τα IDs από την αρχή μέχρι το τέλος
    answer_token_ids = inputs.input_ids[0, answer_start_index:answer_end_index]

    # 4. Decoding (IDs -> String)
    answer = tokenizer.decode(answer_token_ids)
    print(f"Απάντηση: {answer}")
