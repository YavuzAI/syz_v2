import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, ReLU, GlobalAveragePooling2D, Dense, Dropout, Concatenate
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, Callback


class LearningRateLogger(Callback):
    """Custom callback to print learning rate at each epoch."""
    def on_epoch_end(self, epoch, logs=None):
        optimizer = self.model.optimizer
        lr = float(tf.keras.backend.get_value(optimizer.learning_rate))
        print(f"Epoch {epoch+1}: Learning Rate = {lr:.6f}")


def create_model(input_shape=(299, 299, 3), train_last_n_conv_layers=3):
    """
    Enhanced CNN model using InceptionV3 with fine-tuning, additional conv layers, and metadata input.
    
    Args:
        input_shape (tuple): Shape of the input images.
        train_last_n_conv_layers (int): Number of last Conv2D layers to unfreeze.
    
    Returns:
        Compiled Keras model.
    """
    # Image input
    img_input = Input(shape=input_shape, name="image_input")
    metadata_input = Input(shape=(1,), name="metadata_input")
    
    # Load InceptionV3 without top layers (pretrained on ImageNet)
    base_model = InceptionV3(include_top=False, input_tensor=img_input, pooling=None)
    
    # Step 1: Freeze all layers
    for layer in base_model.layers:
        layer.trainable = False

    '''
    # Step 2: Find the last `train_last_n_conv_layers` Conv2D layers and unfreeze them
    conv_layers = [layer for layer in base_model.layers if isinstance(layer, tf.keras.layers.Conv2D)]
    
    # Make sure we don't exceed the number of Conv layers available
    num_conv_layers = len(conv_layers)
    unfreeze_layers = conv_layers[-train_last_n_conv_layers:]  # Select the last 'n' Conv2D layers

    for layer in unfreeze_layers:
        layer.trainable = True

    # Debugging: Print trainable layers
    print("Trainable Conv Layers:")
    for layer in unfreeze_layers:
        print(layer.name, "-", layer.__class__.__name__)
    '''
    
    # Step 3: Modify architecture to ensure model output before GAP is a Conv layer
    last_conv_output = base_model.output  # The last layer of InceptionV3
    x = Conv2D(256, (1, 1), activation='relu', name="custom_last_conv")(last_conv_output)
    x = BatchNormalization()(x)

    # Global Average Pooling
    x = GlobalAveragePooling2D()(x)
    
    # Concatenating metadata (SliceThickness)
    x = Concatenate()([x, metadata_input])
    
    # Dense layers before classification
    x = Dense(16, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(4, activation='relu')(x)
    x = Dropout(0.3)(x)
    
    # Classification Head
    output = Dense(1, activation='sigmoid')(x)
    
    # Define model
    model = Model(inputs=[img_input, metadata_input], outputs=output)
    
    # Compile the model
    optimizer = Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model, base_model


def get_lr_scheduler():
    """
    Returns ReduceLROnPlateau scheduler for dynamic learning rate adjustment.
    """
    return ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6
    )
