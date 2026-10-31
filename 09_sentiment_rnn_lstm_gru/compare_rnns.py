# Compare three kinds of recurrent networks on movie-review sentiment (IMDB):
#   SimpleRNN - the basic RNN; forgets things from long ago (vanishing gradient)
#   LSTM      - has a "cell state" and 3 gates (forget, input, output) to remember longer
#   GRU       - a lighter version of LSTM with 2 gates (reset, update)
# Each model reads a review word by word and says positive (1) or negative (0).
# Run with:  python compare_rnns.py

from tensorflow import keras

VOCAB = 10000   # only keep the 10,000 most common words
MAX_LEN = 150   # cut/pad every review to 150 words

# Reviews come already converted to lists of word numbers
(X_train, y_train), (X_test, y_test) = keras.datasets.imdb.load_data(num_words=VOCAB)
X_train = keras.utils.pad_sequences(X_train, maxlen=MAX_LEN)
X_test = keras.utils.pad_sequences(X_test, maxlen=MAX_LEN)

RNN_LAYERS = {
    "SimpleRNN": keras.layers.SimpleRNN,
    "LSTM": keras.layers.LSTM,
    "GRU": keras.layers.GRU,
}


def build(layer_class):
    model = keras.Sequential([
        keras.Input(shape=(MAX_LEN,)),
        keras.layers.Embedding(VOCAB, 32),  # each word number -> a learned vector of 32 numbers
        layer_class(32),                    # reads the 150 word vectors one after another
        keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


# Turn our own sentence into word numbers the same way the IMDB data was made
word_index = keras.datasets.imdb.get_word_index()


def encode(sentence):
    ids = [1]  # 1 = "start of review" marker
    for word in sentence.lower().split():
        i = word_index.get(word.strip(".,!?"), VOCAB) + 3  # IMDB shifts ids by 3
        ids.append(i if i < VOCAB else 2)                  # 2 = unknown word
    return keras.utils.pad_sequences([ids], maxlen=MAX_LEN)


my_reviews = [
    "this movie was wonderful and the acting was brilliant",
    "a boring and terrible film i want my money back",
]

results = {}
for name, layer_class in RNN_LAYERS.items():
    print(f"\n=== Training {name} ===")
    model = build(layer_class)
    model.fit(X_train, y_train, epochs=2, batch_size=128, validation_split=0.1, verbose=2)
    _, acc = model.evaluate(X_test, y_test, verbose=0)
    results[name] = acc
    for review in my_reviews:
        score = model.predict(encode(review), verbose=0)[0][0]
        print(f"  '{review[:40]}...' -> {'positive' if score > 0.5 else 'negative'} ({score:.2f})")

print("\n=== Test accuracy ===")
for name, acc in results.items():
    print(f"{name:>10}: {acc:.3f}")
