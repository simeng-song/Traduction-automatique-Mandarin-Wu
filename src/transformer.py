import marimo

__generated_with = "0.16.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import os
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    from tensorflow.keras import layers
    from tensorflow.keras.layers import TextVectorization
    return TextVectorization, layers, os, pd, tf


@app.cell
def _(os):
    DATA_DIR = "/home/sibel/Langue-wu/Data/Corpus_aligné"
    TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
    DEV_PATH = os.path.join(DATA_DIR, "dev.csv")
    TEST_PATH = os.path.join(DATA_DIR, "test.csv")

    src = "mandarin"  # langue source (Mandarin)
    tgt = "wu" # langue cible (Wu)
    return DEV_PATH, TEST_PATH, TRAIN_PATH, src, tgt


@app.cell
def _(tf):
    BATCH_SIZE     = 64
    EPOCHS         = 20
    MAX_SRC_LEN    = 50
    MAX_TGT_LEN    = 50
    MAX_VOCAB_SIZE = 4000

    D_MODEL     = 128
    N_ENC       = 4
    N_DEC       = 4
    N_HEADS     = 8
    DFF         = 512
    DROP        = 0.1

    PAD_ID = 0
    UNK_ID = 1
    START_ID = 2
    END_ID = 3

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
        df[src] = df[src].astype(str).str.strip()
        df[tgt] = df[tgt].astype(str).str.strip()
        return df

    train_df = load_data(TRAIN_PATH)
    dev_df   = load_data(DEV_PATH)
    test_df  = load_data(TEST_PATH)

    train_df.head()
    return dev_df, test_df, train_df


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""TextVectorization""")
    return


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
        standardize=None,
        split="character" # par caractère
    )

    tgt_vectorizer = TextVectorization(
        max_tokens=MAX_VOCAB_SIZE,
        output_mode="int",
        output_sequence_length=MAX_TGT_LEN,  # pour decoder shift
        standardize=None,
        split="character",
    )

    # adapter que le train
    src_vectorizer.adapt(train_df[src].values)
    tgt_vectorizer.adapt(train_df[tgt].values)

    SRC_VOCAB_SIZE = len(src_vectorizer.get_vocabulary())
    TGT_VOCAB_SIZE = len(tgt_vectorizer.get_vocabulary()) + 2 # pour START et END

    SRC_VOCAB_SIZE, TGT_VOCAB_SIZE
    return SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, src_vectorizer, tgt_vectorizer


@app.cell
def _(
    AUTOTUNE,
    BATCH_SIZE,
    MAX_TGT_LEN,
    TGT_VOCAB_SIZE,
    dev_df,
    src,
    src_vectorizer,
    test_df,
    tf,
    tgt,
    tgt_vectorizer,
    train_df,
):
    def make_dataset(df, shuffle=True):
        ds = tf.data.Dataset.from_tensor_slices((df[src].values, df[tgt].values))

        def _prep(x_src, x_tgt):
            # vectorize
            x_src_vec = src_vectorizer(x_src)
            x_tgt_vec = tgt_vectorizer(x_tgt)

            x_tgt_vec = x_tgt_vec[x_tgt_vec != 0]

            #start id = size -2 et end id = size - 1
            start_token = tf.constant([TGT_VOCAB_SIZE - 2], dtype=tf.int64)
            end_token = tf.constant([TGT_VOCAB_SIZE - 1], dtype=tf.int64)

            x_tgt_full = tf.concat([start_token, x_tgt_vec, end_token], axis=0)
        
            # decoder inputs
            dec_in  = x_tgt_full[:-1]
            # decoder outputs
            dec_out = x_tgt_full[1:]
        
            dec_in = dec_in[:MAX_TGT_LEN]
            dec_out = dec_out[:MAX_TGT_LEN]
        
            dec_in  = tf.pad(dec_in,  [[0, MAX_TGT_LEN - tf.shape(dec_in)[0]]])
            dec_out = tf.pad(dec_out, [[0, MAX_TGT_LEN - tf.shape(dec_out)[0]]])

            return {
                "encoder_inputs": x_src_vec,
                "decoder_inputs": dec_in
            }, dec_out

        if shuffle:
            ds = ds.shuffle(len(df), reshuffle_each_iteration=True)
    
        return (
            ds.map(_prep, num_parallel_calls=AUTOTUNE)
            .batch(BATCH_SIZE)
            .prefetch(AUTOTUNE)
        )

    train_ds = make_dataset(train_df, shuffle=True)
    dev_ds = make_dataset(dev_df, shuffle=False)
    test_ds = make_dataset(test_df, shuffle=False)

    train_ds
    return dev_ds, train_ds


@app.cell
def _(layers, tf):
    class PositionalEmbedding(layers.Layer):
        def __init__(self, vocab_size, d_model, max_len):
            super().__init__()
            self.d_model = d_model
            self.token_emb = layers.Embedding(vocab_size, d_model)
            self.pos_emb = layers.Embedding(max_len, d_model)

        def call(self, x):
            seq_len = tf.shape(x)[1]
            positions = tf.range(start=0, limit=seq_len, delta=1)
            pos_embeddings = self.pos_emb(positions)
            token_embeddings = self.token_emb(x)
            return token_embeddings + pos_embeddings
    return (PositionalEmbedding,)


@app.cell
def _(mo):
    mo.md(r"""https://keras.io/guides/functional_api/""")
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
    # Inputs
    encoder_inputs = tf.keras.Input(
        shape=(MAX_SRC_LEN,), dtype="int64", name="encoder_inputs"
    )
    decoder_inputs = tf.keras.Input(
        shape=(MAX_TGT_LEN,), dtype="int64", name="decoder_inputs"
    )

    # Encoder
    x_enc = PositionalEmbedding(SRC_VOCAB_SIZE, D_MODEL, MAX_SRC_LEN)(encoder_inputs)
    x_enc = layers.Dropout(DROP)(x_enc)

    for i in range(N_ENC):
        # Self-Attention
        attn_out = layers.MultiHeadAttention(
            num_heads=N_HEADS, 
            key_dim=D_MODEL // N_HEADS,
            dropout=DROP
        )(x_enc, x_enc)
        attn_out = layers.Dropout(DROP)(attn_out)
        x_enc = layers.LayerNormalization(epsilon=1e-6)(x_enc + attn_out)
    
        # Feed-Forward
        ffn_out = layers.Dense(DFF, activation="relu")(x_enc)
        ffn_out = layers.Dense(D_MODEL)(ffn_out)
        ffn_out = layers.Dropout(DROP)(ffn_out)
        x_enc = layers.LayerNormalization(epsilon=1e-6)(x_enc + ffn_out)

    encoder_outputs = x_enc

    # Decoder
    x_dec = PositionalEmbedding(TGT_VOCAB_SIZE, D_MODEL, MAX_TGT_LEN)(decoder_inputs)
    x_dec = layers.Dropout(DROP)(x_dec)

    for i in range(N_DEC):
        # Masked Self-Attention
        masked_attn = layers.MultiHeadAttention(
            num_heads=N_HEADS,
            key_dim=D_MODEL // N_HEADS,
            dropout=DROP
        )(x_dec, x_dec, use_causal_mask=True)
        masked_attn = layers.Dropout(DROP)(masked_attn)
        x_dec = layers.LayerNormalization(epsilon=1e-6)(x_dec + masked_attn)
    
        # Cross-Attention
        cross_attn = layers.MultiHeadAttention(
            num_heads=N_HEADS,
            key_dim=D_MODEL // N_HEADS,
            dropout=DROP
        )(x_dec, encoder_outputs, encoder_outputs)
        cross_attn = layers.Dropout(DROP)(cross_attn)
        x_dec = layers.LayerNormalization(epsilon=1e-6)(x_dec + cross_attn)
    
        # Feed-Forward
        ffn_out = layers.Dense(DFF, activation="relu")(x_dec)
        ffn_out = layers.Dense(D_MODEL)(ffn_out)
        ffn_out = layers.Dropout(DROP)(ffn_out)
        x_dec = layers.LayerNormalization(epsilon=1e-6)(x_dec + ffn_out)

    # Output layer
    decoder_outputs = layers.Dense(TGT_VOCAB_SIZE, name="output_layer")(x_dec)

    # Model
    model = tf.keras.Model(
        inputs=[encoder_inputs, decoder_inputs],
        outputs=decoder_outputs,
        name="Transformer_Mandarin_to_Wu"
    )

    model.summary()
    return (model,)


@app.cell
def _(model, tf):
    # Masked loss
    def masked_loss(y_true, y_pred):
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction='none'
        )
        loss = loss_fn(y_true, y_pred)
    
        # Mask padding tokens (ID = 0)
        mask = tf.cast(y_true != 0, tf.float32)
        loss = loss * mask
    
        return tf.reduce_sum(loss) / tf.reduce_sum(mask)

    # Accuracy metric
    def masked_accuracy(y_true, y_pred):
        pred_ids = tf.cast(tf.argmax(y_pred, axis=-1), y_true.dtype)
        mask = tf.cast(y_true != 0, tf.float32)
    
        correct = tf.cast(tf.equal(y_true, pred_ids), tf.float32)
        correct = correct * mask
    
        return tf.reduce_sum(correct) / tf.reduce_sum(mask)

    # Learning rate schedule
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=5e-4,
        decay_steps=1000,
        decay_rate=0.96,
        staircase=True
    )

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=lr_schedule,
        clipnorm=1.0  # Gradient clipping
    )

    model.compile(
        optimizer=optimizer,
        loss=masked_loss,
        metrics=[masked_accuracy]
    )
    return


@app.cell
def _(EPOCHS, dev_ds, model, tf, train_ds):
    # Callbacks
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        "/home/sibel/Langue-wu/models/best_transformer_m2w.keras",
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )

    early_stop_cb = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )

    # Training
    history = model.fit(
        train_ds,
        validation_data=dev_ds,
        epochs=EPOCHS,
        callbacks=[checkpoint_cb, early_stop_cb]
    )
    return


@app.cell
def _(model):
    model.save("/home/sibel/Langue-wu/models/baseline_tr_m2w.keras")
    return


@app.cell
def _(MAX_TGT_LEN, TGT_VOCAB_SIZE, model, src_vectorizer, tf, tgt_vectorizer):
    def translate(sentence, max_len=MAX_TGT_LEN):

        # START END ID
        start_id = TGT_VOCAB_SIZE - 2
        end_id = TGT_VOCAB_SIZE - 1
    
        # input
        src_seq = src_vectorizer([sentence])
    
        # initialiser decoder
        dec_seq = [start_id]
    
        for _ in range(max_len):
            # decoder input
            dec_input = dec_seq + [0] * (MAX_TGT_LEN - len(dec_seq))
            dec_input = tf.constant([dec_input], dtype=tf.int64)
        
            # prediction
            predictions = model({
                "encoder_inputs": src_seq,
                "decoder_inputs": dec_input
            }, training=False)
        
            # next token
            next_token_logits = predictions[0, len(dec_seq) - 1, :]
            next_token = tf.argmax(next_token_logits).numpy()
        
            # si next token est PAD ou END
            if next_token == end_id or next_token == 0:
                break
        
            dec_seq.append(int(next_token))
    
        # id vers caractere
        tgt_vocab_list = tgt_vectorizer.get_vocabulary()
        result = []
    
        for token_id in dec_seq[1:]:  # sauter START token
            if token_id < len(tgt_vocab_list):
                char = tgt_vocab_list[token_id]
                if char and char != '':
                    result.append(char)
    
        return "".join(result)
    return (translate,)


@app.cell
def _(translate):
    print( translate("你好") )
    print( translate("你今天吃饭了吗") )
    print( translate("我不会说上海话") )
    print( translate("今天天气真好") )
    print( translate("请你帮我一下") )
    print( translate("祝您开心") )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
