import torch
import torch.nn as nn
from torch.nn import functional as F

# Wir laden die Vokabelgröße aus unserem Daten-Skript (95 Zeichen)
# Für diesen Schritt setzen wir sie hart auf 95, basierend auf Phase 1
vocab_size = 95 

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # Jedes Token liest direkt die Logits für das nächste Token aus einer Tabelle
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx und targets sind beide (B, T) Tensoren von Ganzzahlen
        logits = self.token_embedding_table(idx) # (Batch, Zeit, Kanäle/Vocab_size) -> (B, T, C)
        
        if targets is None:
            loss = None
        else:
            # PyTorch erwartet für die Cross-Entropy den Kanal (C) als zweite Dimension
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx ist ein (B, T) Array von Indizes im aktuellen Kontext
        for _ in range(max_new_tokens):
            # Vorhersage generieren
            logits, loss = self(idx)
            # Fokus nur auf den letzten Zeitschritt richten
            logits = logits[:, -1, :] # Wird zu (B, C)
            # Softmax anwenden, um Wahrscheinlichkeiten zu erhalten
            probs = F.softmax(logits, dim=-1) # (B, C)
            # Aus der Wahrscheinlichkeitsverteilung das nächste Zeichen ziehen (Schnitt)
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # Das gezogene Zeichen an den bestehenden Kontext anhängen
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

# --- KLEINER TESTLAUF DES NETZWERKS ---
if __name__ == "__main__":
    # Modell initialisieren
    model = BigramLanguageModel(vocab_size)
    print("Neuronales Netzwerk erfolgreich initialisiert!")
    
    # Einen leeren Startkontext erstellen (Index 0 steht meist für ein Leerzeichen oder Zeilenumbruch)
    context = torch.zeros((1, 1), dtype=torch.long)
    
    # Die KI auffordern, 50 Zeichen zu generieren, während sie noch absolut ungebildet ist
    generated_indices = model.generate(context, max_new_tokens=50)[0].tolist()
    print("\nErster Textgenerierungs-Versuch der ungebildeten KI:")
    print("--------------------------------------------------")
    # Da wir data_prep hier nicht importieren, wandeln wir die Indizes für den Test kurz grob in Text um
    # (Wir sehen uns im nächsten Schritt das echte Zusammenspiel an)
    print(generated_indices)
    print("--------------------------------------------------")
