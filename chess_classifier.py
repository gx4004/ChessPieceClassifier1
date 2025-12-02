import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2, VGG16, ResNet50, DenseNet121
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import ssl, certifi

ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

data_dir = "Chess"
img_size = (224, 224)
batch_size = 16
epochs = 10

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    validation_split=0.2
)

train_gen = train_datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical",
    subset="training",
    shuffle=True
)

val_gen = val_datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

num_classes = train_gen.num_classes
class_names = list(train_gen.class_indices.keys())

def build_head(base_model, dense_units=256, dropout_rate=0.3):
    base_model.trainable = False
    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dense(dense_units, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs=base_model.input, outputs=outputs)
    model.compile(optimizer="adam",
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model

def mobilenet_model():
    base = MobileNetV2(
        input_shape=(img_size[0], img_size[1], 3),
        include_top=False,
        weights="imagenet"
    )
    return build_head(base, dense_units=128, dropout_rate=0.3)

def vgg_model():
    base = VGG16(
        input_shape=(img_size[0], img_size[1], 3),
        include_top=False,
        weights="imagenet"
    )
    return build_head(base, dense_units=256, dropout_rate=0.3)

def resnet_model():
    base = ResNet50(
        input_shape=(img_size[0], img_size[1], 3),
        include_top=False,
        weights="imagenet"
    )
    return build_head(base, dense_units=256, dropout_rate=0.4)

def densenet_model():
    base = DenseNet121(
        input_shape=(img_size[0], img_size[1], 3),
        include_top=False,
        weights="imagenet"
    )
    return build_head(base, dense_units=256, dropout_rate=0.3)

models = {
    "MobileNetV2": mobilenet_model(),
    "VGG16": vgg_model(),
    "ResNet50": resnet_model(),
    "DenseNet121": densenet_model()
}

histories = {}
val_accs = {}

for name, model in models.items():
    print(f"\n===== Training {name} =====\n")
    h = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        verbose=1
    )
    histories[name] = h
    _, acc = model.evaluate(val_gen, verbose=0)
    val_accs[name] = acc

print("\nValidation accuracy:")
for name, acc in val_accs.items():
    print(f"{name}: {acc:.4f}")

best_name = max(val_accs, key=val_accs.get)
best_model = models[best_name]
print(f"\nBest model: {best_name}")

pred_probs = best_model.predict(val_gen, verbose=0)
y_pred = np.argmax(pred_probs, axis=1)
y_true = val_gen.classes

print("\nClassification report for best model:")
print(classification_report(y_true, y_pred, target_names=class_names))

print("\n===== Model Comparison Table =====\n")
print("{:<15} {:<10}".format("Model", "Val Acc"))
print("-" * 30)
for name, acc in val_accs.items():
    print("{:<15} {:<10.4f}".format(name, acc))
print("-" * 30)
print(f"Best model: {best_name}")

plt.figure(figsize=(7, 5))
for name, h in histories.items():
    plt.plot(h.history["val_accuracy"], label=name)
plt.xlabel("Epoch")
plt.ylabel("Val accuracy")
plt.title("Validation accuracy – model comparison")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()