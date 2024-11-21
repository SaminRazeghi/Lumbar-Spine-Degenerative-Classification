import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.callbacks import EarlyStopping
import random
import psutil

df_path = 'Path_to_metadata.csv'
df = pd.read_csv(df_path)

# Load Data
image_paths = df['file_name'].values  # List of image paths
conditions = df['condition'].values  # List of condition labels
xy_labels = df[['x', 'y']].values  # x and y regression labels
condition_level_value = df['condition_level_value'].values  # condition level values

# Split data into train and test sets
train_paths, val_paths, train_conditions, val_conditions, train_xy, val_xy, train_condition_level, val_condition_level = train_test_split(
    image_paths, conditions, xy_labels, condition_level_value, test_size=0.2, random_state=42
)

# Step 2: Load and preprocess images
def load_and_preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

# Preprocess images for train and validation sets
X_train = np.vstack([load_and_preprocess_image(img) for img in train_paths])
X_val = np.vstack([load_and_preprocess_image(img) for img in val_paths])

# Encode condition labels
train_conditions_encoded = pd.get_dummies(train_conditions).values
val_conditions_encoded = pd.get_dummies(val_conditions).values

train_condition_level_encoded = pd.get_dummies(train_condition_level).values
val_condition_level_encoded = pd.get_dummies(val_condition_level).values

# Step 3: Build the model
input_image = tf.keras.Input(shape=(224, 224, 3))

backbone = EfficientNetB0(include_top=False, input_tensor=input_image, weights='imagenet')
backbone.trainable = False

x = tf.keras.layers.GlobalAveragePooling2D()(backbone.output)

condition_output = tf.keras.layers.Dense(train_conditions_encoded.shape[1], activation='softmax', name='condition')(x)
condition_level_value_output = tf.keras.layers.Dense(train_condition_level_encoded.shape[1], activation='softmax', name='condition_level_value')(x)
xy_output = tf.keras.layers.Dense(2, name='xy')(x)

model = tf.keras.Model(inputs=input_image, outputs=[condition_output, condition_level_value_output, xy_output])

# Compile the model
model.compile(
    optimizer='adam',
    loss={'condition': 'categorical_crossentropy', 'condition_level_value': 'categorical_crossentropy', 'xy': 'mean_squared_error'},
    metrics={'condition': 'accuracy', 'condition_level_value': 'accuracy', 'xy': 'mse'}
)

# Set up early stopping
early_stopping = EarlyStopping(
    monitor='val_loss',  # Monitor validation loss
    patience=3,          # Number of epochs with no improvement after which training will stop
    restore_best_weights= True  # Restore the weights of the best epoch
)

class MemoryCallback(tf.keras.callbacks.Callback):
    def on_batch_end(self, batch, logs={}):
        memory = psutil.Process().memory_info().rss / 1024 / 1024  # in MB
        print(f"\rMemory usage: {memory:.2f} MB", end="")

# Combine all callbacks into a list
callbacks = [
    early_stopping, 
    MemoryCallback()
]


# Step 4: Train the model
history = model.fit(
    X_train,
    {'condition': train_conditions_encoded, 'condition_level_value': train_condition_level_encoded, 'xy': train_xy},
    validation_data=(X_val, {'condition': val_conditions_encoded, 'condition_level_value': val_condition_level_encoded, 'xy': val_xy}),
    epochs=20,
    batch_size=32,
    callbacks=callbacks)
    



# Plotting each loss and accuracy metric separately
plt.figure(figsize=(18, 12))

# Plot the total loss over epochs
plt.subplot(2, 2, 1)
plt.plot(history.history['loss'], label='Total Loss', color='blue')
plt.title('Total Loss during Training')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plot condition loss (classification task)
plt.subplot(2, 2, 2)
plt.plot(history.history['condition_loss'], label='Condition Loss', color='orange')
plt.title('Condition Loss (Classification)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plot condition level value loss (classification task)
plt.subplot(2, 2, 3)
plt.plot(history.history['condition_level_value_loss'], label='Condition Level Value Loss', color='purple')
plt.title('Condition Level Value Loss (Classification)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plot xy loss (regression task)
plt.subplot(2, 2, 4)
plt.plot(history.history['xy_loss'], label='XY Loss', color='green')
plt.title('XY Loss (Regression)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Accuracy plots for classification tasks
plt.figure(figsize=(14, 6))

# Plot condition accuracy (classification)
plt.subplot(1, 2, 1)
plt.plot(history.history['condition_accuracy'], label='Condition Accuracy', color='purple')
plt.title('Condition Accuracy (Classification)')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Plot condition level value accuracy (classification)
plt.subplot(1, 2, 2)
plt.plot(history.history['condition_level_value_accuracy'], label='Condition Level Value Accuracy', color='orange')
plt.title('Condition Level Value Accuracy (Classification)')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()


# Step 6: Evaluate model on validation data
eval_results = model.evaluate(X_val, {'condition': val_conditions_encoded, 'condition_level_value': val_condition_level_encoded, 'xy': val_xy})
print(f"Evaluation on Validation Data: {eval_results}")

# Step 7: Show a few random images with true and predicted labels
num_examples = 4
random_indices = random.sample(range(len(X_val)), num_examples)
predictions = model.predict(X_val)

condition_preds = predictions[0]  # Predicted class probabilities for 'condition'
condition_level_value_preds = predictions[1]  # Predicted class probabilities for 'condition_level_value'
xy_preds = predictions[2]  # Predicted (x, y) values

condition_pred_labels = np.argmax(condition_preds, axis=1)
condition_level_value_pred_labels = np.argmax(condition_level_value_preds, axis=1)



num_examples = 5
plt.figure(figsize=(12, 8))
for i, idx in enumerate(random_indices):
    plt.subplot(2, num_examples//2, i+1)
    
    # Get image and show
    img = X_val[idx]
    plt.imshow(img.astype('uint8'))
    plt.axis('off')
    
    # True labels
    true_condition = val_conditions[idx]
    true_condition_level = val_condition_level[idx]
    true_xy = val_xy[idx]
    
    # Predicted labels
    pred_condition = condition_pred_labels[idx]
    pred_condition_level = condition_level_value_pred_labels[idx]
    pred_xy = xy_preds[idx]
    
    plt.title(f"True: {true_condition}, Pred: {pred_condition}\n"
              f"True XY: ({true_xy[0]:.2f}, {true_xy[1]:.2f}), Pred XY: ({pred_xy[0]:.2f}, {pred_xy[1]:.2f})")
    
    # Overlay points
    plt.scatter(pred_xy[0], pred_xy[1], color='red', marker='x', label='Pred XY')
    plt.scatter(true_xy[0], true_xy[1], color='blue', marker='o', label='True XY')

plt.tight_layout()
plt.show()



