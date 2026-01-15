pip install -r req.txt

# run
python main.py build --audio_dir ../../Music/basshouse --index_dir ./index

# exec
python main.py recommend --index_dir ./index --current '..\..\Music\basshouse\€URO TRA$H - What You Looking For (Extended .mp3' --goal up --top_k 10