import tensorflow as tf

# tf.config.optimizer.set_jit(False)  # jit error fix for text

from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2B0
from tensorflow.keras.callbacks import TensorBoard

from clipkit import gpu_memory_fix
from clipkit.cliplayers import ClipMe
from clipkit.dataset import load_data
from clipkit.utils import (
    CheckpointSaver,
    get_image_model,
    get_text_model,
    suppress_verbose,
)

gpu_memory_fix()


train_id = "../dataset/demo_train.csv"

image_model = get_image_model(model=EfficientNetV2B0, active_layers_image_model=10)

text_model_id = "huawei-noah/TinyBERT_General_4L_312D"  # "distilbert-base-uncased"
text_model, tokenizer = get_text_model(model_id=text_model_id, trainable=True)

ckpt_save_dir = "breed_model"
model_logs_dir = "breed_train_logs"
text_max_len = 12
batch_size = 8
epochs = 30
proj_dim = 512
learning_rate = 5e-5

if __name__ == "__main__":
    suppress_verbose()

    train_data = load_data(
        data_id=train_id,
        tokenizer=tokenizer,
        text_max_len=text_max_len,
        mode="train",
        batch_size=batch_size,
    )

    CLIPME = ClipMe(
        image_model_id=image_model, text_model_id=text_model, proj_dim=proj_dim
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate)
    CLIPME.compile(optimizer=optimizer, jit_compile=True)

    tensorboard_callback = TensorBoard(log_dir=model_logs_dir)
    ckpt_callback = CheckpointSaver(ckpt_dir=ckpt_save_dir, max_to_keep=3, verbose=True)

    print("--" * 30)
    print(" Training Started " + chr(0x1F600))
    print("--" * 30)
    CLIPME.fit(
        train_data,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[ckpt_callback, tensorboard_callback],
    )
