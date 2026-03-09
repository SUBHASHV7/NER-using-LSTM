# Named Entity Recognition

## AIM

To develop an LSTM-based model for recognizing the named entities in the text.

## Problem Statement and Dataset
Build a Named Entity Recognition (NER) model that can automatically identify and classify entities like names of people, locations, organizations, and other important terms from text. The goal is to tag each word in a sentence with its corresponding entity label.

### Dataset Name: ner_dataset.csv

<img width="500" height="741" alt="image" src="https://github.com/user-attachments/assets/dcb64627-2a2c-4a71-bf31-05ef5543d7b8" />


Size: Contains thousands of words grouped into sentences with entity annotations.
## DESIGN STEPS

### STEP 1:

Import necessary libraries.

### STEP 2:
Load dataset , Read and clean the input data.

### STEP 3:
Structure data into sentences with word-tag pairs.

### STEP 4:
Convert words and tags to indices using vocab dictionaries.

### STEP 5:
Pad sequences, convert to tensors, and batch them.

### STEP 6:
Create a model with Embedding, BiLSTM, and Linear layers.

### STEP 7:
Use training data to update model weights with loss and optimizer.

### STEP 8:
Check performance on validation data after each epoch.

### STEP 9:
Display predictions or plot loss curves.

## PROGRAM
### Name: SUBHASH V
### Register Number: 212224240163
```python
class BiLSTMTagger(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim=100, hidden_dim=256):
        super(BiLSTMTagger, self).__init__()

        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim * 2, tagset_size)

    def forward(self, input_ids):
        embeddings = self.word_embeddings(input_ids)
        lstm_out, _ = self.lstm(embeddings)
        tag_space = self.hidden2tag(lstm_out)
        return tag_space

model = BiLSTMTagger(len(words) + 1, len(tags)).to(device)
loss_fn = nn.CrossEntropyLoss(ignore_index=tag2idx["O"])
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training and Evaluation Functions
def train_model(model, train_loader, test_loader, loss_fn, optimizer, epochs=3):
    train_losses, val_losses = [], []
    for epoch in range(epochs):
      model.train()
      total_loss = 0
      for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        optimizer.zero_grad()
        outputs = model(input_ids)
        loss = loss_fn(outputs.view(-1, len(tag2idx)), labels.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
      train_losses.append(total_loss)

      model.eval()
      val_loss = 0
      with torch.no_grad():
        for batch in test_loader:
          input_ids = batch["input_ids"].to(device)
          labels = batch['labels'].to(device)
          outputs = model(input_ids)
          loss = loss_fn(outputs.view(-1, len(tag2idx)), labels.view(-1))
          val_loss += loss.item()
      val_losses.append(val_loss)
      print(f"Epoch {epoch+1}: Train Loss = {total_loss:.4f}, Val Loss = {val_loss:.4f}")          
    return train_losses, val_losses

```
## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot

<img width="576" height="505" alt="image" src="https://github.com/user-attachments/assets/4ebb3afb-8cc3-4ef9-a055-ebb8c877c9a8" />


### Sample Text Prediction

<img width="336" height="446" alt="image" src="https://github.com/user-attachments/assets/7475af61-b6a8-4a07-84c3-f26ae281d02d" />


## RESULT

The BiLSTM NER model achieved good accuracy in identifying entities like persons, locations, and organizations. It showed strong performance on frequent tags, with scope for improvement on rarer ones.
