import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2, VGG16
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

data_dir = "archive"
img_size = (64, 64)
batch_size = 32

datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical",
    subset="training",
    shuffle=True
)

val_gen = datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

num_classes = train_gen.num_classes
class_names = list(train_gen.class_indices.keys())

def simple_cnn():
    m = keras.Sequential([
        layers.Input(shape=(img_size[0], img_size[1], 3)),
        layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPooling2D(2),
        layers.Conv2D(64, 3, activation="relu"),
        layers.MaxPooling2D(2),
        layers.Conv2D(128, 3, activation="relu"),
        layers.MaxPooling2D(2),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax")
    ])
    m.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return m

def mobilenet_model():
    base = MobileNetV2(
        input_shape=(img_size[0], img_size[1], 3),
        include_top=False,
        weights=None
    )
    x = layers.Flatten()(base.output)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dense(num_classes, activation="softmax")(x)
    m = keras.Model(inputs=base.input, outputs=x)
    m.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return m

def vgg_model():
    base = VGG16(
        input_shape=(img_size[0], img_size[1], 3),
        include_top=False,
        weights=None
    )
    x = layers.Flatten()(base.output)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dense(num_classes, activation="softmax")(x)
    m = keras.Model(inputs=base.input, outputs=x)
    m.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return m

models = {
    "Simple CNN": simple_cnn(),
    "MobileNetV2": mobilenet_model(),
    "VGG16": vgg_model()
}

histories = {}
val_accs = {}

for name, model in models.items():
    print(f"\n===== Training {name} =====\n")
    h = model.fit(
        train_gen,
        epochs=8,
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