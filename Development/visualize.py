# %%
# Standardlib
from pathlib import Path
# Third party
import pandas as pd
import numpy as np
import plotly.express as px
from wordcloud import WordCloud
from matplotlib import pyplot as plt
import librosa
import librosa.display

test_soundscapes_path = Path(r"Data\test_soundscapes\soundscape_453028782.ogg")
train_metadata_path = Path(r"Data\train_metadata.csv")
# %%
# read csv
train_metadata = pd.read_csv(train_metadata_path)
print(train_metadata.head())
# %%
# distribtution
name_list = pd.unique(train_metadata["scientific_name"])
np.random.seed(1)
colorize = pd.DataFrame(
    data={
        "name": name_list,
        "color": [
            "#" + ''.join(
                [np.random.choice(list("0123456789ABCDEF")) for _ in range(6)]
            ) for _ in range(len(name_list))
        ]
    }
)
df = pd.merge(
    train_metadata,
    colorize,
    left_on="scientific_name",
    right_on="name"
)
fig = px.scatter_geo(
    data_frame=df,
    lat="latitude", lon="longitude",
    opacity=0.8,
    color="color",
    hover_name="common_name",
    projection="orthographic",
    scope="world"
)
fig.update_layout(
    title={
        "text": "<b>Distribution</b>",
        "font": {
            "size": 22,
            "color": "grey"
        },
        "x": 0.5,
        "y": 0.95,
    },
    margin_l=5,
    margin_b=15,
    width=700,
    height=600
)
fig.show()
# Rating
fig = px.histogram(df, x="rating")
fig.show()
# %%
# Common name frequency
wordcloud = WordCloud(
    background_color="white",
    width=1200,
    height=800,
    stopwords=[],
    font_path=r"C:\\Windows\\Fonts\\yumin.ttf"
).generate(
    " ".join(train_metadata["common_name"])
)
plt.figure(figsize=(12, 9))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()
# %%
# 波形とメルスペクトログラム
waveform, sample_rate = librosa.load(test_soundscapes_path)
feature_melspec = librosa.feature.melspectrogram(y=waveform, sr=sample_rate)
feature_mfcc = librosa.feature.mfcc(y=waveform, sr=sample_rate)
plt.figure(figsize=(12, 7))
# 波形
plt.title("wave form")
librosa.display.waveshow(waveform, sr=sample_rate, color='blue')
plt.show()
# メルスペクトログラム
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.title("mel spectrogram")
librosa.display.specshow(
    librosa.power_to_db(feature_melspec, ref=np.max),
    sr=sample_rate,
    x_axis='time',
    y_axis='hz'
)
plt.colorbar(format='%+2.0f dB')
# MFCC
plt.subplot(1, 2, 2)
plt.title("MFCC")
librosa.display.specshow(feature_mfcc, sr=sample_rate, x_axis='time')
plt.colorbar()
plt.show()
# audio
""" display(display.Audio(waveform, rate=sample_rate)) """
# %%
