import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AdamW
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

class BertHandler:
    def __init__(self, train_df, val_df, review_columns, batch_size=16, max_length=512, learning_rate=2e-5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Initialize tokenizer and model
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.models = {
            col: DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2).to(self.device)
            for col in review_columns
        }
        self.optimizers = {
            col: AdamW(self.models[col].parameters(), lr=learning_rate)
            for col in review_columns
        }
        self.loss_fn = torch.nn.CrossEntropyLoss()

        self.batch_size = batch_size
        self.max_length = max_length
        self.train_df = train_df
        self.val_df = val_df
        self.review_columns = review_columns

        # Prepare data loaders for each review column
        self.train_dataloaders = {
            col: self.create_dataloader(train_df, col, "train")
            for col in review_columns
        }
        self.val_dataloaders = {
            col: self.create_dataloader(val_df, col, "validation")
            for col in review_columns
        }

    def tokenize_and_encode(self, df, review_column):
        input_ids = []
        attention_masks = []
        labels = []

        # Tokenization with progress bar
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Tokenizing {review_column}"):
            encoded = self.tokenizer.encode_plus(
                row[review_column],
                add_special_tokens=True,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )

            input_ids.append(encoded['input_ids'])
            attention_masks.append(encoded['attention_mask'])
            labels.append(int(row['recommended']))

        # Convert lists to tensors
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        labels = torch.tensor(labels)

        return input_ids, attention_masks, labels

    def create_dataloader(self, df, review_column, dataset_type):
        # Tokenize and encode the dataset
        input_ids, attention_masks, labels = self.tokenize_and_encode(df, review_column)

        # Create TensorDataset
        dataset = TensorDataset(input_ids, attention_masks, labels)

        # Create DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=(dataset_type == "train")
        )

        return dataloader

    def train(self, epochs=3):
        results = {}
        for col in self.review_columns:
            print(f"Training for column: {col}")

            # Set model and optimizer for the current column
            model = self.models[col]
            optimizer = self.optimizers[col]

            for epoch in range(epochs):
                print(f"Epoch {epoch + 1}/{epochs} for {col}")

                model.train()
                total_loss = 0

                for batch in tqdm(self.train_dataloaders[col], desc=f"Training {col}"):
                    input_ids = batch[0].to(self.device)
                    attention_masks = batch[1].to(self.device)
                    labels = batch[2].to(self.device)

                    optimizer.zero_grad()

                    outputs = model(input_ids, attention_mask=attention_masks, labels=labels)
                    loss = outputs.loss
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()

                avg_train_loss = total_loss / len(self.train_dataloaders[col])
                print(f"Average training loss for {col}: {avg_train_loss}")

            # Evaluate and store results for each column
            metrics = self.evaluate(col)
            results[col] = metrics

        # Compare results
        self.compare_performance(results)

    def evaluate(self, col):
        self.models[col].eval()
        val_loss = 0
        predictions, true_labels = [], []

        for batch in tqdm(self.val_dataloaders[col], desc=f"Validating {col}"):
            input_ids = batch[0].to(self.device)
            attention_masks = batch[1].to(self.device)
            labels = batch[2].to(self.device)

            with torch.no_grad():
                outputs = self.models[col](input_ids, attention_mask=attention_masks, labels=labels)
                loss = outputs.loss
                logits = outputs.logits

            val_loss += loss.item()
            predictions.extend(torch.argmax(logits, dim=1).cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions)
        recall = recall_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions)
        conf_matrix = confusion_matrix(true_labels, predictions)
        avg_val_loss = val_loss / len(self.val_dataloaders[col])

        print(f"\nValidation Metrics for {col}:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        print(f"  Confusion Matrix:\n{conf_matrix}")
        print(f"  Validation Loss: {avg_val_loss:.4f}")

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "conf_matrix": conf_matrix,
            "val_loss": avg_val_loss
        }

    def compare_performance(self, results):
        print("\nComparison of Performance Metrics:")
        for col, metrics in results.items():
            print(f"Column: {col}")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  F1-Score: {metrics['f1']:.4f}")
            print(f"  Validation Loss: {metrics['val_loss']:.4f}")
            print(f"  Confusion Matrix:\n{metrics['conf_matrix']}\n")
