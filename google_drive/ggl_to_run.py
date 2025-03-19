# Step 1: Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Step 2: Load Data
image_df = read_image_df()
metadata_df = read_metadata_df()

# Step 3: Train the Model
model, base_model = create_model(input_shape=(299, 299, 3), train_last_n_conv_layers=3)
lr_scheduler = get_lr_scheduler()
lr_logger = LearningRateLogger()

history = model.fit(
    [train_images, train_metadata], train_labels,
    validation_data=([val_images, val_metadata], val_labels),
    batch_size=16,
    epochs=10,
    callbacks=[lr_scheduler, lr_logger]
)

# Step 4: Save Metrics to Drive
save_model_info(model, history.history, epochs=10, learning_rates=[0.001, 0.0001])
