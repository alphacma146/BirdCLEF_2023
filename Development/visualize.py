# %%
# Standardlib
from pathlib import Path
# Third party
import pandas as pd
import numpy as np
import plotly.express as px
from wordcloud import WordCloud
from matplotlib import pyplot as plt
import torchaudio

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
    # template="plotly_dark",
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
    height=600,

)
# pip install nbformat
fig.show()
# pip install kaleido
fig.write_image("distribution.svg")
fig.write_html("distribution.html")
# Rating
fig = px.histogram(df, x="rating")
fig.show()
fig.write_image("rating.svg")
fig.write_html("rating.html")
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
audio, sample_rate = torchaudio.load(test_soundscapes_path)
spectrogram = torchaudio.transforms.MelSpectrogram(
    sample_rate=sample_rate,
    n_fft=2048,
    win_length=1024,
    center=True,
    pad_mode="reflect",
    power=2.0,
    norm="slaney",
    mel_scale="htk"
)(audio).mean(axis=0)
spectrogram = torchaudio.transforms.AmplitudeToDB()(spectrogram)
spectrogram = spectrogram - spectrogram.min()
spectrogram = spectrogram / spectrogram.max()
plt.figure(figsize=(12, 7))
# 波形
plt.title("wave form")
plt.plot(audio.t().numpy())
plt.show()
# メルスペクトログラム
plt.figure(figsize=(12, 7))
plt.title("mel spectrogram")
plt.imshow(spectrogram.t().numpy(), cmap="hsv", aspect=0.01)
# %%
# audio
""" display(display.Audio(waveform, rate=sample_rate)) """
# %%
