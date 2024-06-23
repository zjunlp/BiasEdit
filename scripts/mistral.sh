python main.py \
    data=stereoset \
    model=mistral_last12 \
    editor=malmen \
    data.n_edits=64 \
    data.batch_size=16 \
    model_device=cuda:0 \
    editor_device=cuda:1 \
    early_stop_patience=5

python main.py \
    data=stereoset \
    model=mistral_last12 \
    editor=malmen \
    data.n_edits=64 \
    data.batch_size=16 \
    model_device=cuda:0 \
    editor_device=cuda:1 \
    editor.load_checkpoint=True \
    eval_only=True \
    data.valid_path=dataset/stereoset/gender_test.json


python main.py \
    data=stereoset \
    model=mistral_last12 \
    editor=malmen \
    data.n_edits=64 \
    data.batch_size=16 \
    model_device=cuda:0 \
    editor_device=cuda:1 \
    editor.load_checkpoint=True \
    eval_only=True \
    data.valid_path=dataset/stereoset/race_test.json

python main.py \
    data=stereoset \
    model=mistral_last12 \
    editor=malmen \
    data.n_edits=64 \
    data.batch_size=16 \
    model_device=cuda:0 \
    editor_device=cuda:1 \
    editor.load_checkpoint=True \
    eval_only=True \
    data.valid_path=dataset/stereoset/religion_test.json