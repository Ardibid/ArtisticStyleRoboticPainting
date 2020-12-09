#!/usr/bin/env python

""" A basic test study on the affordances of StreamLit to develop a dashboard/UI for my ML toolkit"""

__author__ = "Ardavan Bidgoli"
__copyright__ = "Copyright 2020, The ML Toolmaker study"
__credits__ = ["Ardavan Bidgoli"]
__license__ = "MIT"
__version__ = "0.1"
__maintainer__ = "Ardavan Bidgoli"
__email__ = "abidgoli@andrew.cmu.edu"
__status__ = "Prototype study"

# basic ML imports
import tensorflow.keras as keras
import tensorflow as tf 
import numpy as np

# UI/interaction
import streamlit as st 
# import hiplot as hip
import altair as alt
from altair import datum
import pandas as pd
# misc
import copy
import os.path as path

########################################
### Custom functions
########################################

def vae_reconstruction_loss(y_true, y_pred):
    r_loss = K.mean(K.square(y_true-y_pred), axis= [1,2,3])
    return r_loss* r_loss_factor

def vae_kl_loss(y_true, y_pred):
    kl_loss = -0.5*K.sum(1+log_var-K.square(mu)-K.exp(log_var), axis = 1)
    return kl_loss

def vae_loss(y_true, y_pred):
    r_loss  = vae_reconstruction_loss(y_true, y_pred)
    kl_loss = vae_kl_loss(y_true, y_pred)
    return r_loss + kl_loss    


# @st.cache(allow_output_mutation=True)
# or alternatively this:
@st.cache (allow_output_mutation=True)
def load_tf_model():
    model_path = "./weights/"
    main_model_name = "vae.h5"
    encoder_model_name = "encoder.h5"
    decoder_model_name = "decoder.h5"


    main_model = keras.models.load_model(path.join(model_path + main_model_name), custom_objects={'vae_reconstruction_loss': vae_reconstruction_loss, 
                                                                'vae_kl_loss': vae_kl_loss,
                                                                'vae_loss': vae_loss,})
    encoder_model = keras.models.load_model(path.join(model_path + encoder_model_name))
    decoder_model = keras.models.load_model(path.join(model_path + decoder_model_name))

    return main_model,encoder_model, decoder_model

@st.cache
def load_brush_strokes():
    brush_stroke_raw_data = np.load("../data_brushstrokes/brush_strokes_32x64_vae.npy")
    brush_stroke_raw_data = np.expand_dims(brush_stroke_raw_data, axis= 3)
    brush_stroke_raw_data -= 1
    brush_stroke_raw_data *= -1
    return brush_stroke_raw_data

brush_stroke_raw_data = load_brush_strokes()
"""
main_model, encoder_model, decoder_model = load_tf_model()


st.write("# Simple UI for brushstrokes")
st.write(main_model.name)


sample_index = st.slider("Sample index", 0 , 500, 41)
sample = copy.deepcopy(brush_stroke_raw_data[sample_index])
sample_expanded = np.expand_dims(sample, axis= 0)

# latent encoding
latent_vector = encoder_model.predict(sample_expanded)
latent_vector =latent_vector.reshape(latent_vector.shape[1:])
latent_vector = np.clip(latent_vector, 0 ,1)
latent_vector_x = np.arange(8)
# reconstruction
reconstructed_sample = main_model.predict(sample_expanded)
reconstructed_sample =reconstructed_sample.reshape(reconstructed_sample.shape[1:])
reconstructed_sample = np.clip(reconstructed_sample, 0 ,1)

col1, col2, col3 = st.beta_columns(3)

with col1:
    st.image (sample, use_column_width=True)
    st.write("Original image")


with col2:
    pass
with col3:
    st.image (reconstructed_sample, use_column_width=True)
    st.write("Reconstruction")

#st.write(latent_vector)
source = pd.DataFrame({
    'x' : latent_vector_x,
    'latent x' : latent_vector
        })
plot = alt.Chart(source).mark_line().encode(x='x', y = 'latent x')
st.write(plot)
"""

