# Fine-Tuning BERT for Named Entity Recognition (NER)
[![Ask DeepWiki](https://devin.ai/assets/askdeepwiki.png)](https://deepwiki.com/Muhammad-Hassan-Farid/Fine-Tuning-BERT-for-NER)

This repository contains a Jupyter Notebook that demonstrates how to fine-tune a pre-trained BERT model for Named Entity Recognition (NER). The project uses the Hugging Face `transformers`, `datasets`, and `evaluate` libraries to load the CoNLL-2003 dataset, preprocess the data, train the model, and evaluate its performance.

## Table of Contents
- [Introduction to Token Classification](#introduction-to-token-classification)
- [Setup](#setup)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Model Training](#model-training)
- [Saving the Model](#saving-the-model)
- [Inference](#inference)

## Introduction to Token Classification
Token classification is a natural language processing task that assigns a label to each token (word or subword) in a sentence. This project focuses on a specific token classification task:

**Named Entity Recognition (NER):** This involves identifying and categorizing key entities in a text, such as persons, locations, and organizations. The CoNLL-2003 dataset uses the following labels in the IOB (Inside, Outside, Beginning) format:
- `O`: The token does not belong to any entity.
- `B-PER`/`I-PER`: The token is the beginning of/inside a person's name.
- `B-ORG`/`I-ORG`: The token is the beginning of/inside an organization's name.
- `B-LOC`/`I-LOC`: The token is the beginning of/inside a location name.
- `B-MISC`/`I-MISC`: The token is the beginning of/inside a miscellaneous entity.

## Setup
Install the necessary Python libraries using pip:

```bash
pip install transformers datasets tokenizers seqeval evaluate -q
```

## Dataset
This project uses the `conll2003` dataset, which is a standard benchmark for NER. We load it directly from the Hugging Face Hub.

```python
import datasets

conll2003 = datasets.load_dataset("hgissbkh/conll2003-en", split=None)

print(conll2003)
# DatasetDict({
#     train: Dataset({
#         features: ['words', 'ner'],
#         num_rows: 14042
#     })
#     validation: Dataset({
#         features: ['words', 'ner'],
#         num_rows: 3252
#     })
#     test: Dataset({
#         features: ['words', 'ner'],
#         num_rows: 3454
#     })
# })

print(conll2003["train"][0])
# {'words': ['EU', 'rejects', 'German', 'call', 'to', 'boycott', 'British', 'lamb', '.'], 'ner': [4, 0, 8, 0, 0, 0, 8, 0, 0]}

```
The `ner` feature is a sequence of class labels, where the mapping is as follows:
```python
print(conll2003["train"].features["ner"])
# List(ClassLabel(names=['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']))
```

## Data Preprocessing
BERT uses a subword tokenizer (`bert-base-uncased`), which means a single word might be split into multiple tokens. This creates a misalignment between the input tokens and the original labels. To address this, we define a function to tokenize the text and align the labels correctly.

The key steps in the `tokenize_and_align_labels` function are:
1.  Tokenize the input words.
2.  Use `word_ids()` to map each token back to its original word index.
3.  Assign the label `-100` to special tokens (like `[CLS]`, `[SEP]`) and subsequent subword tokens. The CrossEntropyLoss function in PyTorch ignores inputs with the label `-100`, so they are not included in the loss calculation during training.
4.  Assign the correct label to the first token of each word.

```python
from transformers import BertTokenizerFast

tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

def tokenize_and_align_labels(examples, label_all_tokens=True):
    tokenized_inputs = tokenizer(examples["words"], truncation=True, is_split_into_words=True)
    labels = []
    for i, label in enumerate(examples["ner"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                # Set -100 for special tokens
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                # Set label for the first token of a word
                label_ids.append(label[word_idx])
            else:
                # Set -100 for subsequent subword tokens
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# Apply the function to the entire dataset
tokenized_datasets = conll2003.map(tokenize_and_align_labels, batched=True)
```

## Model Training

### 1. Load Model
We load the `bert-base-uncased` model configured for token classification with the number of labels in our dataset (9).

```python
from transformers import AutoModelForTokenClassification

model = AutoModelForTokenClassification.from_pretrained("bert-base-uncased", num_labels=9)
```

### 2. Configure Training
We set up the training arguments and a data collator that will handle batching and padding.

```python
from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification

args = TrainingArguments(
    "test-ner",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

data_collator = DataCollatorForTokenClassification(tokenizer)
```

### 3. Define Metrics
We define a function to compute evaluation metrics (precision, recall, F1-score, and accuracy) using the `seqeval` library. This function correctly handles the `-100` labels by stripping them before evaluation.

```python
import numpy as np
import evaluate

metric = evaluate.load("seqeval")
label_list = conll2003["train"].features["ner"].feature.names

def compute_metrics(eval_preds):
    pred_logits, labels = eval_preds
    pred_logits = np.argmax(pred_logits, axis=2)

    # Remove ignored index (-100) and convert predictions to strings
    predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(pred_logits, labels)
    ]
    # Convert true labels to strings
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(pred_logits, labels)
    ]
    results = metric.compute(predictions=predictions, references=true_labels)
    
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }
```

### 4. Train the Model
We instantiate the `Trainer` with all the components and start the training process.

```python
trainer = Trainer(
   model,
   args,
   train_dataset=tokenized_datasets["train"],
   eval_dataset=tokenized_datasets["validation"],
   data_collator=data_collator,
   tokenizer=tokenizer,
   compute_metrics=compute_metrics
)

trainer.train()
```

## Saving the Model
After training, the fine-tuned model and its tokenizer are saved to disk. We also update the model's configuration file with label mappings for easier use later.

```python
import json

# Save model and tokenizer
model.save_pretrained("ner_model")
tokenizer.save_pretrained("tokenizer")

# Create label mappings
id2label = {str(i): label for i, label in enumerate(label_list)}
label2id = {label: str(i) for i, label in enumerate(label_list)}

# Update and save config
config = json.load(open("ner_model/config.json"))
config["id2label"] = id2label
config["label2id"] = label2id
json.dump(config, open("ner_model/config.json", "w"))
```

## Inference
Finally, we can load our fine-tuned model and use it for inference on new text via the Hugging Face `pipeline`.

```python
from transformers import pipeline

# Load the fine-tuned model from the saved directory
model_fine_tuned = AutoModelForTokenClassification.from_pretrained("ner_model")

# Create a NER pipeline
nlp = pipeline("ner", model=model_fine_tuned, tokenizer=tokenizer)

# Example sentence
example = "Bill Gates is the Founder of Microsoft"
ner_results = nlp(example)
print(ner_results)
```

**Output:**
```
[{'entity': 'I-PER', 'score': 0.99752456, 'index': 1, 'word': 'bill', 'start': 0, 'end': 4}, 
 {'entity': 'I-PER', 'score': 0.99678826, 'index': 2, 'word': 'gates', 'start': 5, 'end': 10}, 
 {'entity': 'I-ORG', 'score': 0.9599028, 'index': 7, 'word': 'microsoft', 'start': 29, 'end': 38}]