# emotion_detection_cpc
Emotion detection in audio utilising self-supervised representations trained with Contrastive Predictive Coding (CPC).

```
# Install dependencies
virtualenv -p python3.7 venv
source venv/bin/activate
make deps
make jupyter

# Download data
wget https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip data/Audio_Speech_Actors_01-24.zip
unzip data/Audio_Speech_Actors_01-24.zip
```
