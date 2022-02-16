# emotion_detection_cpc
This repo provides the code for an emotion recognition system using speech as an input. The performance is boosted using self-supervised representations trained with Contrastive Predictive Coding (CPC). Results have improved from a baseline of 71% to 80% accuracy when using CPC which is a significant relative reduction in error of 30%.

Blog here: https://medium.com/speechmatics/boosting-emotion-recognition-performance-in-speech-using-cpc-ce6b23a05759


## Initial setup 
### Install dependencies
```
virtualenv -p python3.7 venv
source venv/bin/activate
make deps
```

### Download data
```
wget https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip $HOME/RAVDESS/Audio_Speech_Actors_01-24.zip
unzip $HOME/RAVDESS/Audio_Speech_Actors_01-24.zip
```

### Create train, val and test datasets
```
./parse_emotion_dataset.py -j -d $HOME/RAVDESS/Audio_Speech_Actors_01-24 -o ./data
```


## Run CPC pretraining example
```
python3.7 -m cpc.train \
    --train_data=$librispeech_path/train.dbl \
    --val_data=$librispeech_path/val.dbl \
    --expdir=$HOME/exp/cpc_fbank \
    --features_in=fbank \
    --batch_size=16 \
    --window_size=128 \
    --steps=500000 \
    --hidden_size=512 \
    --out_size=256 \
    --timestep=12
```


## Run emotion recognition example
### Training
```
python3.7 -m emotion_id.train \
    --expdir=$HOME/exp/emotion_id \
    --cpc_path=$HOME/exp/cpc_fbank/model.pt \
    --train_data=data/train.dbl \
    --val_data=data/val.dbl \
    --window_size=1024 \
    --model=rnn_bi \
    --batch_size=8 \
    --steps=40000 \
    --lr=1e-4 \
    --hidden_size=512 \
    --dropout_prob=0.1 \
    --emotion_set_path=data/emotion_set.txt
```

### Decoding and scoring
```
python3.7 -m emotion_id.decode \
    --eval_file_path=data/test.dbl \
    --cpc_path=$HOME/exp/cpc_fbank/model.pt \
    --model_path=$HOME/exp/emotion_id/model.pt \
    --output_dir=$HOME/exp/emotion_id/predictions \
    --window_size=1024 \

python3.7 -m emotion_id.score \
    --emotion_set_path=data/emotion_set.txt \
    --ref data/test.dbl \
    --pred $HOME/exp/emotion_id/predictions/score.dbl \
    --output $HOME/exp/emotion_id/score

grep "accuracy" $HOME/exp/emotion_id/score/score_results.json
```

---

License: [MIT](LICENSE.txt)
