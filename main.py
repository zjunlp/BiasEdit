import hydra
from omegaconf import DictConfig, OmegaConf

import importlib, logging

from data.base import make_loader
from model import make_model

import wandb

LOG = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config: DictConfig):
    
    if config.use_wandb:
        wandb.init(    
            project = f"{config.data.name}_{config.model.name_or_path.split('/')[-1]}",
            name = f"{config.editor.name}_{str(config.data.n_edits)}",
            config = OmegaConf.to_container(config, resolve = True)
        )

    LOG.info(f"\n\n{OmegaConf.to_yaml(config)}\n")
    
    data_module = importlib.import_module(f"data.{config.data.name.lower()}")
    data_class = getattr(data_module, f"{config.data.name}Dataset")
    train_loader, valid_loader =  make_loader(config, data_class)
    
    model = make_model(config.model).to(config.model_device)

    editor_module = importlib.import_module(f"editor.{config.editor.name}")
    editor_class = getattr(editor_module, config.editor.name.upper())
    editor = editor_class(config, model)

    if config.eval_only:
        editor.valid(valid_loader)
    else:
        editor.run(train_loader, valid_loader)
    
if __name__ == "__main__":
    main()