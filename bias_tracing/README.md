# Bias Tracing

Trace bias effect in states of language model.

## Tracing
Run the scripts `bash scripts/gpt2m.sh`.

Results are saved in `./results`.

## Histograms
```shell
>>> python fig.py -h
    usage: fig.py [-h] [--root ROOT] [--num_layer NUM_LAYER] [--model_name MODEL_NAME] [--bias {gender,race}] [--num_sample NUM_SAMPLE]

    optional arguments:
    -h, --help            show this help message and exit
    --root ROOT           the path of results
    --num_layer NUM_LAYER
                            The num of model layers.
    --model_name MODEL_NAME
                            The model name.
    --bias {gender,race}  The bias type.
    --num_sample NUM_SAMPLE
                            The num of samples
```


Thanks for the original code from [*ROME*](https://github.com/kmeng01/rome).