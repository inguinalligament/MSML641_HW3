################################################################################################
####        TITLE: MSML HW3                                                                 ####
####        DESCRIPTION: SENTIMENT ANALYSIS - MODELS.PY                                     ####
####        AUTHOR: BRADLEY SCOTT                                                           ####
####        UMD ID: 119 775 028                                                             ####
####        DATE: 26OCT2025                                                                 ####
####        REFERENCES USED (see paper for full details):                                   ####
####            ChatGPT 5                                                                   ####
################################################################################################

'''
[BS10262025] mod3_641_000001
[BS10262025] import all necessary modules
'''
from tensorflow.keras import layers, models, optimizers

'''
[BS10262025] mod3_641_000005
[BS10262025] Build out a function to build the models
    NB: Required functionality as follows:
       architecture: RNN, LSTM, Bidirectional LSTM
       activation functions: Sigmoid, RelU, Tanh
       optimizer: Adam, SGD, RMSProp
       Sequence length: done in preprocess.py pad_data function
       Stability: No strategy vs. Gradient Clipping (clipnorm and clipvalue)
       Include an embedding layer (size: 100).
       Use 2 hidden layers (hidden size: 64).
       Use dropout (0.3–0.5) to reduce overfitting.
       Batch size: 32.
       Use a fully connected output layer with a sigmoid activation for binary classification.
       Use binary cross-entropy loss.
       Fix all other hyperparameters when varying one factor (e.g., only change the optimizer, keep architecture and sequence length fixed).
'''
def build_model(
    arch,
    vocab_size,
    seq_len,
    emb_dim=100,              # embedding layer size fixed to 100
    hidden_units=64,          # hidden layer size fixed to 64
    dropout_rate=0.3,         # dropout between 0.3 and 0.5
    optimizer_name='adam',
    activation = 'tanh', # RNN/LSTM cell activation
    stability='none',         # 'none' | 'clipnorm' | 'clipvalue'
    clipnorm=1.0,
    clipvalue=None
):
    # Input layer
    inp = layers.Input(shape=(seq_len,))
    x = layers.Embedding(input_dim=vocab_size, output_dim=emb_dim)(inp)

    # Recurrent hidden layers
    if arch == 'rnn':
        x = layers.SimpleRNN(
            hidden_units, return_sequences=True,
            dropout=dropout_rate, recurrent_dropout=dropout_rate,
            activation=activation
        )(x)
        x = layers.SimpleRNN(
            hidden_units,
            dropout=dropout_rate, recurrent_dropout=dropout_rate,
            activation=activation
        )(x)

    elif arch == 'lstm':
        x = layers.LSTM(
            hidden_units, return_sequences=True,
            dropout=dropout_rate, recurrent_dropout=dropout_rate,
            activation=activation,                # vary this
            recurrent_activation='sigmoid'       # keep default
        )(x)
        x = layers.LSTM(
            hidden_units,
            dropout=dropout_rate, recurrent_dropout=dropout_rate,
            activation=activation,
            recurrent_activation='sigmoid'
        )(x)

    elif arch == 'bilstm':
        x = layers.Bidirectional(layers.LSTM(
            hidden_units, return_sequences=True,
            dropout=dropout_rate, recurrent_dropout=dropout_rate,
            activation=activation,
            recurrent_activation='sigmoid'
        ))(x)
        x = layers.Bidirectional(layers.LSTM(
            hidden_units,
            dropout=dropout_rate, recurrent_dropout=dropout_rate,
            activation=activation,
            recurrent_activation='sigmoid'
        ))(x)

    else:
        raise ValueError(f"Unknown architecture: {arch}")

    # Fully connected output layer with sigmoid
    out = layers.Dense(1, activation='sigmoid')(x)
    model = models.Model(inputs=inp, outputs=out)

    # Optional gradient clipping
    opt_kwargs = {}
    if stability == 'clipnorm':
        opt_kwargs['clipnorm'] = float(clipnorm)
    elif stability == 'clipvalue':
        if clipvalue is None:
            raise ValueError("clipvalue must be provided when stability='clipvalue'")
        opt_kwargs['clipvalue'] = float(clipvalue)

    # Choose optimizer
    if optimizer_name == 'adam':
        opt = optimizers.Adam(**opt_kwargs)
    elif optimizer_name == 'sgd':
        opt = optimizers.SGD(learning_rate=1e-2, momentum=0.9, nesterov=True, **opt_kwargs)
    elif optimizer_name == 'rmsprop':
        opt = optimizers.RMSprop(**opt_kwargs)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    # Compile model with binary crossentropy
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model
