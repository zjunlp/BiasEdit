python main.py \
    data=stereoset \
    model=gpt2m_last123 \
    editor=malmen \
    data.n_edits=128 \
    data.batch_size=64 \
    model_device=cuda:1 \
    editor_device=cuda:0 \
    editor.loc_coef=1.0 \
    editor.lr=1e-7 \
    editor.meta_lr=1e-6

python main.py \
    data=stereoset \
    model=gpt2m_last123 \
    editor=malmen \
    data.n_edits=128 \
    data.batch_size=64 \
    model_device=cuda:1 \
    editor_device=cuda:0 \
    editor.load_checkpoint=True \
    eval_only=True \
    data.valid_path=dataset/stereoset/gender_test.json \
    editor.loc_coef=1.0 \
    editor.lr=1e-7 \
    editor.meta_lr=1e-6


python main.py \
    data=stereoset \
    model=gpt2m_last123 \
    editor=malmen \
    data.n_edits=128 \
    data.batch_size=64 \
    model_device=cuda:1 \
    editor_device=cuda:0 \
    editor.load_checkpoint=True \
    eval_only=True \
    data.valid_path=dataset/stereoset/race_test.json \
    editor.loc_coef=1.0 \
    editor.lr=1e-7 \
    editor.meta_lr=1e-6

python main.py \
    data=stereoset \
    model=gpt2m_last123 \
    editor=malmen \
    data.n_edits=128 \
    data.batch_size=64 \
    model_device=cuda:1 \
    editor_device=cuda:0 \
    editor.load_checkpoint=True \
    eval_only=True \
    data.valid_path=dataset/stereoset/religion_test.json \
    editor.loc_coef=1.0 \
    editor.lr=1e-7 \
    editor.meta_lr=1e-6
