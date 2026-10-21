import torch

# 1. Den Text laden
with open('Faust.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print(f"Länge des Textes in Zeichen: {len(text)}")

# 2. Alle einzigartigen Zeichen finden (das Alphabet des Modells)
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"Einzigartige Zeichen ({vocab_size}): {''.join(chars)}")

# 3. Zeichen in Zahlen übersetzen und umgekehrt (Das Wörterbuch)
char_to_int = { ch:i for i,ch in enumerate(chars) }
int_to_char = { i:ch for i,ch in enumerate(chars) }

# Encoder: Nimmt Text und gibt eine Liste von Zahlen zurück
encode = lambda s: [char_to_int[c] for c in s]
# Decoder: Nimmt eine Liste von Zahlen und gibt Text zurück
decode = lambda l: ''.join([int_to_char[i] for i in l])

# Kleiner Test zur Kontrolle
test_text = "Hallo Faust"
print(f"\nTest-Text: {test_text}")
encoded_test = encode(test_text)
print(f"In Zahlen: {encoded_test}")
print(f"Zurück übersetzt: {decode(encoded_test)}")

# 4. Den gesamten Faust in Zahlen übersetzen und als PyTorch-Tensor speichern
data = torch.tensor(encode(text), dtype=torch.long)
print(f"\nGesamte Datenform (PyTorch Tensor): {data.shape}")

# 5. Daten in Training (90%) und Validierung (10%) aufteilen
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

print(f"Training-Tokens: {len(train_data)}")
print(f"Validierung-Tokens: {len(val_data)}")

# --- NEUER CODE FÜR PHASE 2 ---

# Festlegen der Hyperparameter für die Häppchengröße
block_size = 8  # Kontext-Länge: Wie viele Zeichen liest die KI, um das nächste vorherzusagen?
batch_size = 4  # Batch-Größe: Wie viele unabhängige Häppchen verarbeiten wir gleichzeitig?

torch.manual_seed(1337) # Sorgt dafür, dass die "Zufallszahlen" immer gleich sind (wichtig zum Lernen)

def get_batch(split):
    # Wählt aus, ob wir Trainings- oder Validierungsdaten nutzen
    data_source = train_data if split == 'train' else val_data
    
    # Generiert zufällige Startpunkte im Text
    ix = torch.randint(len(data_source) - block_size, (batch_size,))
    
    # Packt die Eingabesequenzen (X) und die Zielsequenzen (Y) zusammen
    x = torch.stack([data_source[i:i+block_size] for i in ix])
    y = torch.stack([data_source[i+1:i+block_size+1] for i in ix])
    return x, y

# Testlauf: Ein Paket aus den Trainingsdaten generieren
xb, yb = get_batch('train')
print("\n=== BATCH-TEST ===")
print(f"Eingabe (X) Form: {xb.shape}")
print(f"Ziel (Y) Form:    {yb.shape}")

print("\nExplizites Beispiel für das erste Häppchen im Batch:")
for t in range(block_size):
    context = xb[0, :t+1]
    target = yb[0, t]
    print(f"Wenn die KI sieht: {context.tolist()} (Text: '{decode(context.tolist())}') -> Solls vorhersehen: {target} ('{decode([target.item()])}')")
