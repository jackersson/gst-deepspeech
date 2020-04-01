
```bash
python3 -m venv venv
source venv/bin/activate

pip install --upgrade wheel pip setuptools
pip install --upgrade --requirement requirements.txt

export GOOGLE_APPLICATION_CREDENTIALS=$PWD/credentials/gs_viewer.json
dvc pull
```

```bash
youtube-dl --format "best[ext=mp4][protocol=https]" https://www.youtube.com/watch?v=qTWODHHGjzs -o data/video.mp4
```

