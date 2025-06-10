import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import (Input, Embedding, Conv1D, GlobalMaxPooling1D, Dense,
                                    Dropout, Concatenate, BatchNormalization, SeparableConv1D, 
                                    Bidirectional, LSTM, Multiply)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
import pickle
import numpy as np
import os
import time
from nltk import pos_tag
# Set up webapp page
st.set_page_config(layout="wide", page_title="Movie Review Sentiment Analysis")

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger_eng')
stop_words = set(nltk.corpus.stopwords.words('english')) - {
    'not', 'no', 'never', 'none', 'nor', 'ain', 'aren', 'couldn', 'didn',
    'doesn', 'don', 'hadn', 'hasn', 'haven', 'isn', 'mightn', 'mustn',
    'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn','it', 'this', 'that', 'is', 'are', 'was', 'were'
}

# Same parameter with model training
MAX_NB_WORDS = 25000  
MAX_SEQUENCE_LENGTH = 450 
FILTER_SIZES = [2, 3, 4, 5, 6] 
EMBEDDING_DIM = 300

#Attention layel
class ChannelAttention(tf.keras.layers.Layer):
    def __init__(self, ratio=8, **kwargs):
        super(ChannelAttention, self).__init__(**kwargs)
        self.ratio = ratio

    def build(self, input_shape):
        channel = input_shape[-1]
        self.shared_dense = tf.keras.layers.Dense(channel // self.ratio,
                                 activation='relu',
                                 kernel_initializer='he_normal',
                                 use_bias=False)
        self.dense1 = tf.keras.layers.Dense(channel, kernel_initializer='he_normal')
        super(ChannelAttention, self).build(input_shape)

    def call(self, x):
        avg_pool = tf.reduce_mean(x, axis=1, keepdims=True)
        max_pool = tf.reduce_max(x, axis=1, keepdims=True)
        avg_out = self.shared_dense(avg_pool)
        max_out = self.shared_dense(max_pool)
        avg_out = self.dense1(avg_out)
        max_out = self.dense1(max_out)
        out = avg_out + max_out
        scale = tf.nn.sigmoid(out)
        return scale

class PositionAwareAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], 1),
                                 initializer='glorot_uniform', trainable=True)
        super().build(input_shape)

    def call(self, x):
        seq_len = tf.shape(x)[1]
        position = tf.range(0, seq_len, dtype=x.dtype)
        position = tf.expand_dims(position, axis=-1)
        pos_att = tf.matmul(x, self.W)
        a = tf.nn.softmax(pos_att, axis=1)
        return tf.reduce_sum(x * a, axis=1)

# Abbreviation dictionary
contractions_dict = {
    "ain't": "are not", "aren't": "are not", "can't": "cannot", "could've": "could have", 
    "couldn't": "could not", "didn't": "did not","doesn't": "does not", "don't": "do not", 
    "hadn't": "had not","hasn't": "has not", "haven't": "have not", "he'd": "he would",
    "he'll": "he will", "he's": "he is", "how'd": "how did","how'll": "how will", 
    "how's": "how is", "i'd": "I would","i'll": "I will", "i'm": "I am", "i've": "I have",
    "isn't": "is not", "it'd": "it would", "it'll": "it will","it's": "it is", 
    "let's": "let us", "might've": "might have","must've": "must have", "mustn't": "must not", 
    "needn't": "need not","she'd": "she would", "she'll": "she will", "she's": "she is",
    "should've": "should have", "shouldn't": "should not", "that's": "that is",
    "they'd": "they would", "they'll": "they will", "they're": "they are",
    "they've": "they have", "wasn't": "was not", "we'd": "we would",
    "we'll": "we will", "we're": "we are", "we've": "we have",
    "weren't": "were not", "what'll": "what will", "what're": "what are",
    "what's": "what is", "what've": "what have", "where's": "where is",
    "who'd": "who would", "who'll": "who will", "who's": "who is",
    "why's": "why is", "won't": "will not", "would've": "would have",
    "wouldn't": "would not", "you'd": "you would", "you'll": "you will",
    "you're": "you are", "you've": "you have"
}

contractions_re = re.compile('(%s)' % '|'.join(sorted(contractions_dict.keys(), key=lambda x: (-len(x), x))))
def expand_contractions(text, contractions_dict=contractions_dict):
    def replace(match):
        return contractions_dict[match.group(0)]
    return contractions_re.sub(replace, text)

def enhanced_negation_handling(text):
    text = text.lower()
    patterns = [
        (r"\b(can)(not)\b", r"\1 \2"),
        (r"\b(don't)\b", "do not"),
        (r"\b(doesn't)\b", "does not"),
        (r"\b(is|was|were)\s+not\b", "not"),
        (r"\bno\s+(\w+)\b", r"not have \1"),
        (r"\black\s+of\b", "not have"),
        (r"\bwon't\b", "will not"),
        (r"\bwouldn't\b", "would not"),
        (r"\b(n't)\b", " not"),
        (r"\bnot\s+have\s+(\w+)\b", r"not_have_\1")
    ]
    for _ in range(2):
        for pat, repl in patterns:
            text = re.sub(pat, repl, text)
    text = re.sub(r"not_have_(\w+)", r"not have \1", text)
    return text

def identify_aspect_terms(text):
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    aspect_terms = []
    current_chunk = []

    # Recognize only certain types of noun phrases
    for word, pos in tagged:
        if pos.startswith(('NN', 'JJ')): 
            current_chunk.append(word)
        elif current_chunk:
            term = ' '.join(current_chunk)
            if len(term) > 3 and term.lower() not in ['it', 'this', 'that', 'is', 'are', 'was', 'were']:
                aspect_terms.append(term)
            current_chunk = []

    if current_chunk:
        term = ' '.join(current_chunk)
        if len(term) > 3 and term.lower() not in ['it', 'this', 'that']:
            aspect_terms.append(term)

    return list(set(aspect_terms))

# Data preprocess same with the model training
def data_processing(text):
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = enhanced_negation_handling(text)
    text = expand_contractions(text)
    
    aspect_terms = identify_aspect_terms(text)
    aspect_terms = sorted(aspect_terms, key=len, reverse=True)
    for term in aspect_terms:
        if term in text and f"[ASPECT]{term}[/ASPECT]" not in text:
            text = text.replace(term, f"[ASPECT]{term}[/ASPECT]")
    
    text = re.sub(r"[^a-zA-Z\s\[\]]", ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)
    filtered = [
        w for w in tokens
        if w not in stop_words or w in {'not', 'no', 'never'}
    ]
    return ' '.join(filtered)

# Data AdvancedLemmatizer same with the model training
class AdvancedLemmatizer:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.verb_past_mapping = {
            'broken': 'break', 'believable': 'believe', 'appalled': 'appall',
            'made': 'make', 'went': 'go', 'had': 'have', 'did': 'do',
            'said': 'say', 'taken': 'take', 'given': 'give', 'seen': 'see', 'thought': 'think'
        }
        self.film_terms = {
            'filmmaking', 'screenplay', 'cinematography',
            'nonsensical', 'blockbuster', 'award_winning'
        }

    def get_wordnet_pos(self, treebank_tag):
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return None

    def lemmatize(self, text):
        tokens = word_tokenize(text)
        tagged = pos_tag(tokens)
        lemmas = []
        for word, pos in tagged:
            if word.lower() in self.film_terms:
                lemmas.append(word)
                continue
            if word.lower() in self.verb_past_mapping:
                lemmas.append(self.verb_past_mapping[word.lower()])
                continue
            if word.lower() in {'is', 'are', 'was', 'were'}:
                lemmas.append(word)
                continue
            wn_pos = self.get_wordnet_pos(pos) or wordnet.NOUN
            lemma = self.lemmatizer.lemmatize(word, wn_pos)
            if lemma.endswith('e') and word.endswith('ing'):
                lemma = word
            lemmas.append(lemma)
        return ' '.join(lemmas)

advanced_lemmatizer = AdvancedLemmatizer()

# hybrid model same with the training model
def build_bilstm_cnn_model(num_words, tokenizer): 
    input_layer = Input(shape=(MAX_SEQUENCE_LENGTH,))
    
    # ABSA special treatment
    aspect_token_index = tokenizer.word_index.get('[ASPECT]', None) 
    
    embedding_layer = Embedding(
        input_dim=num_words,
        output_dim=EMBEDDING_DIM,
        trainable=True
    )(input_layer)
    
    # Fix ABSA processing code
    if aspect_token_index is not None:
        aspect_embed = tf.keras.layers.Embedding(
            input_dim=1,
            output_dim=EMBEDDING_DIM,
            embeddings_initializer='uniform',
            name='aspect_embed'
        )(tf.zeros_like(input_layer))
        
        aspect_mask = tf.expand_dims(tf.cast(tf.equal(input_layer, aspect_token_index), -1))
        
        embedding_layer = tf.keras.layers.Add()([
            embedding_layer * (1 - tf.cast(aspect_mask, tf.float32)),
            aspect_embed * tf.cast(aspect_mask, tf.float32)
        ])  

    embedding_layer = tf.keras.layers.SpatialDropout1D(0.15)(embedding_layer)

    conv_layer = SeparableConv1D(
        256, 3, activation='relu', padding='same')(embedding_layer)

    # Bilstm with unit 384
    bilstm = Bidirectional(LSTM(
        384,
        return_sequences=True,
        kernel_regularizer=l2(0.001),
        dropout=0.15,
        recurrent_dropout=0.1,
        activation='tanh'
    ))(conv_layer)

    # residual dimension 768
    residual = Conv1D(768, 1, padding='same')(conv_layer)
    bilstm = tf.keras.layers.Add()([bilstm, residual])

    attention_heads = []
    for _ in range(2):
        att = PositionAwareAttention()(bilstm)
        attention_heads.append(att)
    attention = Concatenate()(attention_heads)

    cnn_branches = []
    # convolution branch filters is 128
    for sz in FILTER_SIZES:
        conv = Conv1D(128, sz, activation='elu', padding='same',
                      kernel_regularizer=l2(0.003))(bilstm)
        se = ChannelAttention()(conv)
        conv = Multiply()([conv, se])
        max_pool = GlobalMaxPooling1D()(conv)
        cnn_branches.append(max_pool)

    combined = Concatenate()([attention] + cnn_branches)
    combined = BatchNormalization()(combined)
    combined = Dropout(0.5)(combined)

    # connected layer units to 512
    dense = Dense(512, activation='elu', kernel_regularizer=l2(0.003))(combined)
    dense = BatchNormalization()(dense)
    dense = Dropout(0.45)(dense)

    output = Dense(1, activation='sigmoid', dtype='float32')(dense)

    model = Model(inputs=input_layer, outputs=output)
    return model
  
# loading 
@st.cache_resource
def load_model_and_tokenizer():
    required_files = {
        'tokenizer': 'tokenizer.pkl',
        'best_weights': 'best_model.weights.h5',
        'swa_weights': 'swa_model.weights.h5'
    }
    
    missing = [f for f in required_files.values() if not os.path.exists(f)]
    if missing:
        st.error(f"Missing files: {', '.join(missing)}")
        st.error("Please ensure all required files are in the current directory")
        return None, None, None
    
    try:
        with open(required_files['tokenizer'], 'rb') as f:
            tokenizer = pickle.load(f)
        
        num_words = min(MAX_NB_WORDS, len(tokenizer.word_index)) + 1
        
        model_best = build_bilstm_cnn_model(num_words, tokenizer)  
        model_swa = build_bilstm_cnn_model(num_words, tokenizer) 
        
        model_best.load_weights(required_files['best_weights'])
        model_swa.load_weights(required_files['swa_weights'])
        
        model_best.compile(optimizer='adam', loss='binary_crossentropy')
        model_swa.compile(optimizer='adam', loss='binary_crossentropy')
            
        return model_best, model_swa, tokenizer
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None, None, None

# predict sentiment get from the model training
def predict_sentiment(text, model_best, model_swa, tokenizer):
    try:
        processed = data_processing(text)
        lemmatized = advanced_lemmatizer.lemmatize(processed)
        seq = tokenizer.texts_to_sequences([lemmatized])
        padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
        
        pred_best = model_best.predict(padded, verbose=0)[0][0]
        pred_swa = model_swa.predict(padded, verbose=0)[0][0]
        prediction = (0.7 * pred_best) + (0.3 * pred_swa)
        
        return prediction, processed, lemmatized
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        return 0.5, "", ""

# initiate session state
if 'history' not in st.session_state:
    st.session_state.history = []
    
if 'show_result' not in st.session_state:
    st.session_state.show_result = False
    
if 'current_result' not in st.session_state:
    st.session_state.current_result = None

model_best, model_swa, tokenizer = load_model_and_tokenizer()

# Develop web app
st.markdown("""
<style>
    .main-container {
        max-width: 1200px;
        margin: 0 auto;
    }
    
    .title-container {
        background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
        border-radius: 15px;
        padding: 25px 30px;
        margin-bottom: 30px;
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
        color: white;
    }
    
    .input-container {
        background-color: #ffffff;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.05);
        margin-bottom: 30px;
    }
    
    .result-card {
        background-color: #ffffff;
        border-radius: 15px;
        padding: 18px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.05);
        margin-bottom: 30px;
    }
    
    .progress-container { 
        position: relative; 
        height: 15px; 
        background: #f0f2f6; 
        border-radius: 10px; 
        overflow: hidden;
        margin: 20px 0;
    }
    
    .progress-bar { 
        height: 100%; 
        transition: width 0.5s ease; 
        border-radius: 10px;
    }
    
    .history-container {
        background-color: #ffffff;
        border-radius: 15px;
        padding: 18px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.05);
        position: sticky; 
        top: 20px; 
        max-height: 80vh; 
        overflow-y: auto;
    }
    
    .history-item {
        padding: 15px;
        margin: 10px 0;
        border-radius: 10px;
        background-color: #f8f9fa;
        transition: all 0.3s ease;
        border-left: 5px solid #6a11cb;
        position: relative;
        overflow: hidden;
    }
    
    .history-text-preview {
        max-height: 60px; 
        overflow-y: hidden;
        position: relative;
        margin: 10px 0;
        font-size: 14px;
        color: #555;
        line-height: 1.4;
    }
    
    .expand-btn {
        display: block;
        width: 100%;
        text-align: left;
        padding: 8px 12px;
        margin-top: 8px;
        background-color: #f1f3f5;
        border: none;
        border-radius: 6px;
        cursor: pointer;
        font-size: 13px;
        color: #666;
        transition: all 0.2s;
    }
    
    .expand-btn:hover {
        background-color: #e0e3e7;
    }
  
    .stButton>button {
        border-radius: 50px;
        padding: 10px 25px;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    
    form {
        margin-bottom: 0 !important;
    }
    
   
</style>""", unsafe_allow_html=True)

st.markdown('<div class="main-container">', unsafe_allow_html=True)

st.markdown("""
<div class="title-container">
    <h1 style="margin:0; font-size:36px;">🎬 Movie Review Sentiment Analysis</h1>
    <p style="margin:10px 0 0; font-size:18px; opacity:0.9;">
        Sentiment analysis with model BILSTM+CNN+Attention
    </p>
</div>""", unsafe_allow_html=True)
import re
# Creating a two-column layout
col1, col2 = st.columns([2, 1], gap="large")

with col1:
    with st.form(key='review_form'):
        st.markdown('<div class="form-container">', unsafe_allow_html=True)
        st.subheader("📝 Enter Your Movie Review")
        
        text = st.text_area("", height=200, 
                           placeholder="Example: This movie completely exceeded my expectations...",
                           label_visibility="collapsed",
                           value=st.session_state.get('review_text', ''),
                           key='review_text')
        
        analyze_btn = st.form_submit_button("🚀 Start Analysis", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # validation
    validation_error = None
    if analyze_btn and text.strip():
        if text.strip().isdigit():
            validation_error = "Review cannot contain only numbers!"
        
        special_chars = set('!@#$%^&*()_+-=[]{}|;:\'",.<>?/`~\\')
        
        cleaned_text = ''.join(char for char in text.strip() if char not in special_chars)
        if not cleaned_text:
            validation_error = "Review cannot consist entirely of special characters!"
        
        # Display validation error
        if validation_error:
            st.error(validation_error)
            st.session_state.show_result = False

         # Proceed after validate
        elif model_best is None or model_swa is None or tokenizer is None:
            st.error("Models not loaded. Please check file availability.")
            st.session_state.show_result = False
        else:
            with st.spinner("Analyzing your review..."):
                try:
                    start_time = time.time()
                    prediction, processed, lemmatized = predict_sentiment(text, model_best, model_swa, tokenizer)
                    processing_time = time.time() - start_time
                    
                    confidence = abs(prediction - 0.5) * 2
                    confidence = max(0.0, min(1.0, confidence))
                    
                    if prediction < 0.5:
                        emoji = "😊"
                        label = "Positive reviews"
                        color = "#1f77b4"
                    else:
                        emoji = "😞"
                        label = "Negative reviews"
                        color = "#ff4b4b"
                    
                    st.session_state.current_result = {
                        "emoji": emoji,
                        "label": label,
                        "color": color,
                        "confidence": confidence,
                        "processing_time": processing_time,
                        "processed_text": processed,
                        "lemmatized_text": lemmatized,
                        "raw_text": text
                    }
                    
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    record = {
                        "timestamp": timestamp,
                        "raw_text": text,
                        "sentiment": label,
                        "confidence": f"{confidence:.1%}",
                        "color": color,
                        "emoji": emoji
                    }
                    st.session_state.history.append(record)
                    st.session_state.show_result = True
                    
                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")
                    st.session_state.show_result = False
    
    # display output
    if st.session_state.show_result and st.session_state.current_result:
        result = st.session_state.current_result
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown(f"""
            <div style="display:flex; align-items:center; margin-bottom:20px;">
                <div style="font-size:48px; margin-right:15px;">{result['emoji']}</div>
                <div>
                    <h2 style="margin:0; color:{result['color']};">{result['label']}</h2>
                    <p style="margin:5px 0 0; font-size:18px; color:{result['color']};">
                        Confidence: {result['confidence']:.1%}
                    </p>
                </div>
            </div>""", unsafe_allow_html=True)
        
        progress_width = result['confidence'] * 100
        st.markdown(f"""
            <div class="progress-container">
                <div class="progress-bar" style="width:{progress_width}%; background:{result['color']};"></div>
            </div>""", unsafe_allow_html=True)
        
        st.caption(f"Analysis completed in {result['processing_time']:.2f} seconds")
        
        # with st.expander("View Processing Details", expanded=False):
        #     st.subheader("Original Text")
        #     st.write(result['raw_text'])
            
        #     st.subheader("Processed Text")
        #     st.write(result['processed_text'])
            
        #     st.subheader("Lemmatized Text")
        #     st.write(result['lemmatized_text'])
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with st.expander("System Information", expanded=False):
        st.write("TensorFlow version:", tf.__version__)
        st.write("Working directory:", os.getcwd())
        st.write("Directory files:", [f for f in os.listdir() if f.endswith(('.pkl', '.h5'))])

# History layel
with col2:
    st.markdown('<div class="history-container">', unsafe_allow_html=True)
    st.subheader("📜 Analysis History")
    
    if st.session_state.history:
        history_sorted = sorted(
            st.session_state.history,
            key=lambda x: x["timestamp"],
            reverse=True
        )
        
        for record in history_sorted[:5]:
            tag_class = "positive-tag" if "Positive" in record["sentiment"] else "negative-tag"
            tag_text = "Positive" if "Positive" in record["sentiment"] else "Negative"

            with st.expander(f"{record['emoji']} {tag_text} (Confidence: {record['confidence']})"):
                st.markdown(f"""
                <div class="history-item fade-enter">
                    <div style="font-size:14px; color:#666; margin-bottom:10px;">
                        {record['timestamp']}
                    </div>
                    <div style="margin-top:10px; font-size:14px; color:#555;">
                        {record['raw_text']}
                    </div>
                </div>""", unsafe_allow_html=True)
    else:
        st.info("No analysis history yet. Submit a review to see history here.")
    
    st.markdown('</div>', unsafe_allow_html=True)
