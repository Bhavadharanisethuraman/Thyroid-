import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    accuracy_score,
    precision_recall_curve,
    average_precision_score
)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load the trained model
model = tf.keras.models.load_model('DL.keras')
