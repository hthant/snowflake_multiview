import os, csv
from tensorflow.keras.callbacks import Callback
from datetime import datetime as dt

class CSVLogger(Callback):
    def __init__(self, csv_path):
        super().__init__()
        self.csv_path = csv_path

        with open(self.csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        train_acc = logs.get("acc", logs.get("accuracy"))
        train_loss = logs.get("loss")
        val_acc = logs.get("val_acc", logs.get("val_accuracy"))
        val_loss = logs.get("val_loss")

        with open(self.csv_path, mode='a+', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch + 1, train_loss, train_acc, val_loss, val_acc])


# Defines model checkpoints such that only better performing models based on validation accuracy
# have thier weights saved after each epoch
class CustomModelCheckpoint(Callback):
    def __init__(self, filepath, **kwargs):
        super().__init__()
        self.filepath = filepath
        self.best_val_acc = -float("inf")
        self.kwargs = kwargs

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        # Accessing training and validation metrics
        train_acc = logs.get("acc", logs.get("accuracy"))
        train_loss = logs.get("loss")
        val_acc = logs.get("val_acc", logs.get("val_accuracy"))
        val_loss = logs.get("val_loss")

        # Creating unique filepath with metrics
        current_file_path = self.filepath.format(
            epoch=epoch + 1,
            val_acc=val_acc,
            train_acc=train_acc,
            val_loss=val_loss,
            train_loss=train_loss,
            timestamp=dt.now().strftime("%Y/%m/%d-%H:%M:%S")  # add timestamp for uniqueness
        )
        
        # Save only if validation accuracy improves
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            
            # Try to delete existing file, handle potential errors
            try:
                if os.path.exists(current_file_path):
                    os.remove(current_file_path)
                self.model.save(current_file_path, overwrite=True)
                print(f"\nModel improved and saved to {current_file_path}")
            except Exception as e:
                print(f"\nError during saving: {e}")
