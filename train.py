import torch
import torch.nn as nn
from torch.nn import functional as F

# 1. Hyperparameter für den CPU-Durchbruch (Hier gibt es keine doppelten Einträge mehr!)
batch_size = 64       # Mehr Daten gleichzeitig verarbeiten
block_size = 8        # Kontextlänge
max_iters = 5000      # 5000 Schritte für schnelles CPU-Training
eval_interval = 500   # Anzeige alle 500 Schritte
learning_rate = 5e-3  # Schnellere Korrekturen auf der CPU (0.005)
n_embd = 64           # Verdoppelte neuronale Kapazität für echten Textfortschritt

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Nutze Hardware-Beschleunigung: {device.upper()}")

torch.manual_seed(1337)

# 2. Daten laden (aus der Faust.txt)
with open('Faust.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

char_to_int = { ch:i for i,ch in enumerate(chars) }
int_to_char = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [char_to_int[c] for c in s]
decode = lambda l: ''.join([int_to_char[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data_source = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_source) - block_size, (batch_size,))
    x = torch.stack([data_source[i:i+block_size] for i in ix])
    y = torch.stack([data_source[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(10)
        for k in range(10):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# 3. Das optimierte Modell mit Vergangenheits-Kommunikation
class SmartContextLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx) 
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) 
        x = tok_emb + pos_emb 
        
        # Mathematische Kommunikation über Dreiecksmatrix
        wei = torch.tril(torch.ones(T, T, device=device))
        wei = wei / wei.sum(1, keepdim=True) 
        x = wei @ x 
        
        logits = self.lm_head(x) 
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# 4. Modell initialisieren
model = SmartContextLanguageModel(vocab_size)
m = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print("\nStarte Training mit hochgedrehter Kapazität... Bitte warten...")
print("------------------------------------------------------------------")

# 5. Trainingsschleife
for iter in range(max_iters):

    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"Schritt {iter:4d}: Trainings-Loss {losses['train']:.4f} | Validierungs-Loss {losses['val']:.4f}")

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print("------------------------------------------------------------------")
print("Training abgeschlossen!\n")

print("Generierter Text der trainierten KI:")
print("==================================================")
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=300).tolist()[0]))
print("==================================================")
