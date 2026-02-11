import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from model import GPT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

block_size = 128

# Load tokenizer
tokenizer = Tokenizer.from_file("tokenizer.json")

# Build model
model = GPT(tokenizer.get_vocab_size(), block_size).to(device)

# Load weights
model.load_state_dict(torch.load("mini_gpt_15m_final.pt", map_location=device))
model.eval()

def generate(prompt, max_new_tokens=50, temperature=0.8, top_k=30):
    input_ids = tokenizer.encode(prompt).ids
    input_ids = torch.tensor([input_ids], dtype=torch.long).to(device)

    for _ in range(max_new_tokens):
        input_ids = input_ids[:, -block_size:]

        with torch.no_grad():
            logits = model(input_ids)

        logits = logits[:, -1, :] / temperature

        values, indices = torch.topk(logits, top_k)
        logits_filtered = torch.full_like(logits, float('-inf'))
        logits_filtered.scatter_(1, indices, values)

        probs = F.softmax(logits_filtered, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        input_ids = torch.cat((input_ids, next_token), dim=1)

    return tokenizer.decode(input_ids[0].tolist())

# Interactive loop
while True:
    prompt = input("\nYou: ")
    if prompt.lower() == "exit":
        break
    
    output = generate(prompt)
    print("AI:", output)