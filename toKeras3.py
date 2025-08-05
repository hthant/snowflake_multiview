from typing import Any, Callable, Optional, Tuple
import tensorflow as tf

#Takes images and transforms them into a TensorFlow dataset
class CustomTFDataset(tf.keras.utils.Sequence):
    def __init__(
        self,
        X_generator,
        y_generator,
        *,
        transform: Optional[Callable],
        target_transform: Optional[Callable] = None,
    ) -> None:
        
        self.X_generator = X_generator
        self.y_generator = y_generator
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        return len(self.X_generator)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        X_batch = tf.convert_to_tensor(self.X_generator[idx].data)
        y_batch = tf.convert_to_tensor(self.y_generator[idx].data)
        if self.transform:
            X_batch = self.transform(X_batch)
        if self.target_transform:
            y_batch = self.target_transform(y_batch)
        return X_batch, y_batch

