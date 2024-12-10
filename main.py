from models.init_model import unet
from preproc import X_train, X_val, y_train, y_val
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import tensorflow_addons as tfa

import os
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

#Memory related commands for tensorflow GPU.
tf.keras.mixed_precision.set_global_policy('mixed_float16')
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


# model compilations
unet_ins = unet(input_size=(192,256,3), num_classes=1)
adamw_optimizer = tfa.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4)

# Compile the model
unet_ins.compile(optimizer=adamw_optimizer,
                loss='binary_crossentropy',
                metrics=['accuracy'])

#model training
his = unet_ins.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=2,
    epochs=50,
    shuffle=True
)


unet_ins.save("final.h5")
#unet_ins = load_model("init_model.h5")

# METRICS
val_loss, val_acc = unet_ins.evaluate(X_val, y_val)
print("Validation loss: ", val_loss)
print("Validation accuracy: ", val_acc)

print("\nPredicting with model...\n")
pred_probs = unet_ins.predict(X_val)

#handle probability from network
pred_masks = (pred_probs > 0.5).astype("uint8")

num_masks = len(pred_masks)

for i in range(num_masks):
    fig, axs = plt.subplots(1, 3, figsize=(15,5))

    axs[0].imshow(X_val[i])
    axs[0].set_title("Original Image")
    axs[0].axis('off')

    axs[1].imshow(y_val[i].squeeze(), cmap='gray')
    axs[1].set_title("Ground Truth Mask")
    axs[1].axis('off')

    axs[2].imshow(pred_masks[i].squeeze(), cmap='gray')
    axs[2].set_title("Predicted Mask")
    axs[2].axis('off')

    plt.tight_layout()
    # Save the figure for the current sample
    plt.savefig(f"results/comparison_{i}.png")
    plt.close(fig)  # Close the figure to free memory

y_true_flat = y_val.flatten()
y_pred_flat = pred_masks.flatten()

TP = np.sum((y_true_flat == 1) & (y_pred_flat == 1))
FP = np.sum((y_true_flat == 0) & (y_pred_flat == 1))
FN = np.sum((y_true_flat == 1) & (y_pred_flat == 0))
TN = np.sum((y_true_flat == 0) & (y_pred_flat == 0))

precision = TP / (TP + FP + 1e-7)  # add small epsilon to avoid division by zero
recall = TP / (TP + FN + 1e-7)
f1_score = 2 * (precision * recall) / (precision + recall + 1e-7)

#dice for segmentation
intersection = np.sum((y_true_flat == 1) & (y_pred_flat == 1))
dice = (2 * intersection) / (np.sum(y_true_flat == 1) + np.sum(y_pred_flat == 1) + 1e-7)

print(f"TP: {TP}\t FP: {FP}\t FN: {FN}\t TN: {TN}\n")
print(f"Precision: {precision}\t Recall: {recall}\t F1: {f1_score}\n")
print(f"Dice: {dice}")