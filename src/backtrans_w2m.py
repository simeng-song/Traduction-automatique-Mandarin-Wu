import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import os
    import pandas as pd
    import numpy as np
    import tensorflow as tf

    from tensorflow.keras import layers
    from tensorflow.keras.layers import TextVectorization
    return TextVectorization, layers, os, pd, tf


@app.cell
def _(os):
    DATA_DIR = "/Users/songsimeng/INALCOM2/Langue-wu/Data/Corpus_aligné"
    TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
    DEV_PATH = os.path.join(DATA_DIR, "dev.csv")
    TEST_PATH = os.path.join(DATA_DIR, "test.csv")

    src = "wu"  # langue source
    tgt = "mandarin" # langue cible
    return DEV_PATH, TEST_PATH, TRAIN_PATH, src, tgt


@app.cell
def _(tf):
    BATCH_SIZE     = 64
    EPOCHS         = 20
    MAX_SRC_LEN    = 50
    MAX_TGT_LEN    = 50
    MAX_VOCAB_SIZE = 4000

    D_MODEL     = 128
    N_ENC       = 2
    N_DEC       = 2
    N_HEADS     = 4
    DFF         = 256
    DROP        = 0.1

    #optimisation du pipeline
    AUTOTUNE = tf.data.AUTOTUNE 
    return (
        AUTOTUNE,
        BATCH_SIZE,
        DFF,
        DROP,
        D_MODEL,
        EPOCHS,
        MAX_SRC_LEN,
        MAX_TGT_LEN,
        MAX_VOCAB_SIZE,
        N_DEC,
        N_ENC,
        N_HEADS,
    )


@app.cell
def _(DEV_PATH, TEST_PATH, TRAIN_PATH, pd, src, tgt):
    def load_data(path):
        df = pd.read_csv(path)
        df = df[[src, tgt]].dropna()
        # On force le type str pour éviter les valeurs non-textuelles
        df[src] = df[src].astype(str) 
        df[tgt] = df[tgt].astype(str)
        return df

    train_df = load_data(TRAIN_PATH)
    dev_df   = load_data(DEV_PATH)
    test_df  = load_data(TEST_PATH)

    # On choisit les caractères rares pour que START/END soient bien un seul token
    START = "§" # début de séquence
    END = "¤" # fin de séquence

    # On encadre la phrase cible pour apprendre au décodeur où commencer et finir
    train_df[tgt] = START + train_df[tgt] + END
    dev_df[tgt]   = START + dev_df[tgt] + END
    test_df[tgt]  = START + test_df[tgt] + END
    return END, START, dev_df, train_df


@app.cell
def _(
    MAX_SRC_LEN,
    MAX_TGT_LEN,
    MAX_VOCAB_SIZE,
    TextVectorization,
    src,
    tgt,
    train_df,
):
    src_vectorizer = TextVectorization(
        max_tokens=MAX_VOCAB_SIZE,
        output_mode="int",
        output_sequence_length=MAX_SRC_LEN,
        standardize=None,   # on garde le texte tel quel
        split="character" # pour éviter d'utiliser une segmentation en mots
    )
    tgt_vectorizer = TextVectorization(
        max_tokens=MAX_VOCAB_SIZE,
        output_mode="int",
        output_sequence_length=MAX_TGT_LEN + 1,  # +1 car on fera ensuite un "shift" 
        standardize=None,
        split="character"
    )

    # On construit le vocab seulement sur le train set
    src_vectorizer.adapt(train_df[src].values)
    tgt_vectorizer.adapt(train_df[tgt].values)

    SRC_VOCAB_SIZE = len(src_vectorizer.get_vocabulary())
    TGT_VOCAB_SIZE = len(tgt_vectorizer.get_vocabulary())
    return SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, src_vectorizer, tgt_vectorizer


@app.cell
def _(
    AUTOTUNE,
    BATCH_SIZE,
    dev_df,
    src,
    src_vectorizer,
    tf,
    tgt,
    tgt_vectorizer,
    train_df,
):
    def make_dataset(df):
        ds = tf.data.Dataset.from_tensor_slices((df[src], df[tgt]))

        def _prep(x_src, x_tgt):
            x_src_vec = src_vectorizer(x_src)
            x_tgt_vec = tgt_vectorizer(x_tgt)

            dec_in = x_tgt_vec[:-1] # inputs reçoit tout sauf le dernier token
            dec_out = x_tgt_vec[1:] # outputs reçoit tout sauf le premier token

            return {
                "encoder_inputs": x_src_vec,
                "decoder_inputs": dec_in
            }, dec_out

        # Pipeline: shuffle -> map(prétraitement) -> batch -> prefetch
        return (
            ds.shuffle(len(df))
                .map(_prep,num_parallel_calls=AUTOTUNE)
                .batch(BATCH_SIZE)
                .prefetch(AUTOTUNE)
        )

    train_ds = make_dataset(train_df)
    dev_ds = make_dataset(dev_df)
    return dev_ds, train_ds


@app.cell
def _(layers, tf):
    class PositionalEmbedding(layers.Layer):
        def __init__(self, vocab_size, d_model, max_len):
            super().__init__()
            self.token = layers.Embedding(vocab_size, d_model) # ids -> vecteurs
            self.pos = layers.Embedding(max_len, d_model) # 0, 1, 2, ..., max_len-1

        def call(self, x):
            seq_len = tf.shape(x)[1]
            positions = tf.range(start=0, limit=seq_len, delta=1)
            return self.token(x) + self.pos(positions)
    return (PositionalEmbedding,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Définition du Transformer
    """)
    return


@app.cell
def _(
    DFF,
    DROP,
    D_MODEL,
    MAX_SRC_LEN,
    MAX_TGT_LEN,
    N_DEC,
    N_ENC,
    N_HEADS,
    PositionalEmbedding,
    SRC_VOCAB_SIZE,
    TGT_VOCAB_SIZE,
    layers,
    tf,
):
    # Entrées
    encoder_inputs = tf.keras.Input(shape=(MAX_SRC_LEN,), dtype="int64", name="encoder_inputs")
    decoder_inputs = tf.keras.Input(shape=(MAX_TGT_LEN,), dtype="int64", name="decoder_inputs")

    key_dim = D_MODEL // N_HEADS
    dropout = layers.Dropout(DROP)

    # Encodeur
    # On ajoute les embeddings pour fournir au modèle l'ordre des caractères
    x_enc = PositionalEmbedding(SRC_VOCAB_SIZE, D_MODEL, MAX_SRC_LEN)(encoder_inputs)

    for _ in range(N_ENC):
        attn_out = layers.MultiHeadAttention(num_heads=N_HEADS, key_dim=key_dim)(x_enc, x_enc)
        x_enc = layers.LayerNormalization()(x_enc + dropout(attn_out))

        # Transformation non-linéaire position par position
        ffn_out = layers.Dense(DFF, activation="relu")(x_enc)
        ffn_out = layers.Dense(D_MODEL)(ffn_out)
        x_enc = layers.LayerNormalization()(x_enc + dropout(ffn_out))

    encoder_outputs = x_enc

    # Décodeur
    x_dec = PositionalEmbedding(TGT_VOCAB_SIZE, D_MODEL, MAX_TGT_LEN)(decoder_inputs)
    for _ in range(N_DEC):
        # pour éviter que le décodeur voie le futur
        masked_att = layers.MultiHeadAttention(num_heads=N_HEADS, key_dim=key_dim)(x_dec, x_dec, use_causal_mask=True)
        x_dec = layers.LayerNormalization()(x_dec + dropout(masked_att))

        # Le décodeur peut consulter les représentations de l'encodeur
        cross_att = layers.MultiHeadAttention(num_heads=N_HEADS, key_dim=key_dim)(x_dec, encoder_outputs, encoder_outputs)
        x_dec = layers.LayerNormalization()(x_dec + dropout(cross_att))

        ffn_out = layers.Dense(DFF, activation="relu")(x_dec)
        ffn_out = layers.Dense(D_MODEL)(ffn_out)
        x_dec = layers.LayerNormalization()(x_dec + dropout(ffn_out))

    # Projection finale vers la distribution sur le vocabulaire cible
    decoder_outputs = layers.Dense(TGT_VOCAB_SIZE)(x_dec)
    return decoder_inputs, decoder_outputs, encoder_inputs


@app.cell
def _(decoder_inputs, decoder_outputs, encoder_inputs, tf):
    model = tf.keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.summary()
    return (model,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Fonction de perte masquée
    """)
    return


@app.cell
def _(tf):
    # On ignore les positions (token PAD) pour ne pas pénaliser le modèle
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True,
        reduction="none"
    ) 
    def masked_loss(y_true, y_pred):
        mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
        loss = loss_fn(y_true, y_pred) # loss par token
        loss = tf.reduce_sum(loss * mask) / tf.reduce_sum(mask) # moyenne sur les tokens non-pad
        return loss
    return (masked_loss,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Compilation du modèle
    """)
    return


@app.cell
def _(masked_loss, model, tf):
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    model.compile(
        optimizer=optimizer,
        loss=masked_loss,
        metrics=["sparse_categorical_accuracy"]  # indicatif (token-level)
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Entraînement du modèle
    """)
    return


@app.cell
def _(EPOCHS, dev_ds, model, tf, train_ds):
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True
        )
    ]

    history = model.fit(
        train_ds, 
        validation_data=dev_ds, 
        epochs=EPOCHS, 
        callbacks=callbacks
    )
    return


@app.cell
def _(model):
    model.save("/Users/songsimeng/INALCOM2/Langue-wu/models/backtrans_w2m.keras")
    return


@app.cell
def _(END, MAX_TGT_LEN, START, model, src_vectorizer, tf, tgt_vectorizer):
    def translate(sentence):
        src_seq = src_vectorizer([sentence]) # encodage source

        # Vocabulaire cible et ids des tokens spéciaux
        vocab = tgt_vectorizer.get_vocabulary()
        start_id = int(tgt_vectorizer([START])[0][0].numpy())
        end_id = int(tgt_vectorizer([END])[0][0].numpy())
        # Initialisation avec START
        dec_seq = [start_id]

        for _ in range(MAX_TGT_LEN):
            dec_input = tf.constant([dec_seq + [0] * (MAX_TGT_LEN - len(dec_seq))])
            logits = model({"encoder_inputs": src_seq, "decoder_inputs": dec_input})
            next_id = int(tf.argmax(logits[0, len(dec_seq)-1]))

            # Arrêter si PAD/END
            if next_id == 0 or next_id == end_id:
                break

            dec_seq.append(next_id)

        # Reconstruire la chaîne
        tokens = dec_seq[1:]
        return "".join(vocab[i] for i in tokens)
    return (translate,)


@app.cell
def _(translate):
    # Exemples
    print(translate("阿拉今朝出去吃饭。"))
    print(translate("哪能勿晓得呢？"))
    print(translate("今朝落雨伐？"))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Chargement du corpus wu monolingue
    """)
    return


@app.cell
def _():
    WU_TXT_PATH = "/Users/songsimeng/INALCOM2/Langue-wu/Data/wu_mono.txt"
    with open(WU_TXT_PATH, "r", encoding="utf-8") as f:
        wu_sentences = [line.strip() for line in f if line.strip()]

    print("Nombre de phrases wu :", len(wu_sentences))
    print("Exemples :", wu_sentences[:5])
    return (wu_sentences,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Backtranslation : Wu -> Mandarin
    """)
    return


@app.cell
def _(translate, wu_sentences):
    bt_pairs = []

    for wu in wu_sentences:
        zh = translate(wu)

        # Filtrage minimal 
        if not zh:
            continue
        if len(zh) < 2:
            continue
        if len(zh) > 2 * len(wu):  
            continue

        bt_pairs.append((zh, wu))

    print("Nombre de paires générées :", len(bt_pairs))
    print("Exemple :", bt_pairs[:3])
    return (bt_pairs,)


@app.cell
def _(bt_pairs, pd):
    bt_df = pd.DataFrame(bt_pairs, columns=["mandarin", "wu"])

    BT_OUT_PATH = "/Users/songsimeng/INALCOM2/Langue-wu/Data/backtranslated_w2m.csv"
    bt_df.to_csv(BT_OUT_PATH, index=False, encoding="utf-8")
    return


if __name__ == "__main__":
    app.run()
