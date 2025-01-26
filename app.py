import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from PIL import Image, ImageOps

# === Ծրագրի վերնագիրը ===
st.title("Ձեռագիր թվերի ճանաչում")
st.write("Ծրագիրը օգտագործում է մեքենայական ուսուցման մոդել՝ ձեռագիր թվերի (0-9) դասակարգման համար։")

# === Մոդելի ներբեռնում և պատրաստում ===
@st.cache(allow_output_mutation=True)
def load_model():
    # MNIST տվյալների բազայի ներբեռնում
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train / 255.0  # Նորմալացում
    X_train = X_train.reshape(-1, 28, 28, 1)  # Ձևափոխում CNN-ի համար
    y_train_one_hot = tf.keras.utils.to_categorical(y_train, 10)  # One-hot կոդավորում

    # Մոդելի կառուցում
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    # Մոդելի կոմպիլացում
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Մոդելի ուսուցում
    model.fit(X_train, y_train_one_hot, epochs=3, batch_size=32, verbose=0)
    return model

model = load_model()

# === Օգտագործողի նկարի ներբեռնում ===
uploaded_file = st.file_uploader("Ներբեռնեք ձեռագիր թվի պատկեր (28x28 պիքսել, մոխրագույնի երանգներով)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Պատկերի բացում և մշակում
    image = Image.open(uploaded_file).convert("L")  # Փոխակերպում մոխրագույնի երանգների
    image = ImageOps.invert(image)  # Գույների ինվերսիա (եթե ֆոնը սև է)
    image = image.resize((28, 28))  # Չափսի փոփոխություն 28x28
    image_array = np.array(image) / 255.0  # Նորմալացում

    # Ներբեռնված պատկերի ցուցադրում
    st.image(image, caption="Ներբեռնված պատկեր", width=150)

    # Մոդելի կանխատեսում
    image_array = image_array.reshape(1, 28, 28, 1)
    prediction = model.predict(image_array)
    predicted_label = np.argmax(prediction)

    # Արդյունքի ցուցադրում
    st.subheader(f"Կանխատեսված թիվը՝ {predicted_label}")
    st.write("Մոդելի վստահությունը յուրաքանչյուր թվի համար․")
    st.bar_chart(prediction[0])

# === MNIST-ի պատահական պատկեր ===
if st.button("Ցույց տալ պատահական պատկեր MNIST-ից"):
    (X_train, y_train), _ = mnist.load_data()
    idx = np.random.randint(0, len(X_train))
    sample_image = X_train[idx]
    st.image(sample_image, caption=f"Թիվը՝ {y_train[idx]}", width=150)
