from keras import Input, Model
from keras.layers import concatenate, dot, Embedding, GRU, Dense, Activation


def define_nmt(hidden_size, embedding_size, timesteps, src_vocab_size, tar_vocab_size):
    # trainable layers
    encoder_emb = Embedding(src_vocab_size, embedding_size)
    encoder_gru = GRU(hidden_size, return_sequences=True, return_state=True)
    decoder_emb = Embedding(tar_vocab_size, embedding_size)
    decoder_gru = GRU(hidden_size, return_sequences=True, return_state=True)
    decoder_tan = Dense(hidden_size, activation="tanh")
    decoder_softmax = Dense(tar_vocab_size, activation='softmax')

    def define_encoder(encoder_inputs):
        encoder_embed = encoder_emb(encoder_inputs)
        encoder_out, encoder_state = encoder_gru(encoder_embed)
        return encoder_out, encoder_state

    def define_decoder(decoder_inputs, encoder_states, decoder_init_state):
        decoder_embed = decoder_emb(decoder_inputs)
        decoder_out, decoder_state = decoder_gru(decoder_embed, initial_state=decoder_init_state)
        attention = dot([decoder_out, encoder_states], axes=[2, 2])
        attention = Activation('softmax')(attention)
        context = dot([attention, encoder_states], axes=[2, 1])
        decoder_context = concatenate([context, decoder_out])
        decoder_pred = decoder_tan(decoder_context)
        decoder_pred = decoder_softmax(decoder_pred)
        return decoder_pred, decoder_state

    # joint model for training
    encoder_inputs = Input(shape=(timesteps,))
    decoder_inputs = Input(shape=(timesteps - 1,))
    encoder_out, encoder_state = define_encoder(encoder_inputs)
    decoder_pred, _ = define_decoder(decoder_inputs, encoder_out, encoder_state)
    model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_pred)
    model.compile(optimizer='adam', loss='categorical_crossentropy')

    # encoder inference model
    encoder_inf_inputs = Input(shape=(timesteps,))
    encoder_inf_out, encoder_inf_state = define_encoder(encoder_inf_inputs)
    encoder_model = Model(inputs=encoder_inf_inputs, outputs=[encoder_inf_out, encoder_inf_state])

    # decoder inference model
    encoder_inf_states = Input(shape=(timesteps, hidden_size,))
    decoder_init_state = Input(shape=(hidden_size,))
    decoder_inf_inputs = Input(shape=(1,))
    decoder_inf_pred, decoder_inf_state = define_decoder(decoder_inf_inputs, encoder_inf_states, decoder_init_state)
    decoder_model = Model(inputs=[encoder_inf_states, decoder_init_state, decoder_inf_inputs],
                          outputs=[decoder_inf_pred, decoder_inf_state])

    return model, encoder_model, decoder_model