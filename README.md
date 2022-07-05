# Growing-Neural-Networks---Stage2A
Ce repository Git contient les codes utilisés lors du stage et qui étaient conscis pour être exécutés en Grid5000 

## Papiers de recherche sur les transformers, réseaux de neurones grossissants...

__[Attention is all you need](https://arxiv.org/abs/1706.03762)__

__[BERT](https://arxiv.org/pdf/1810.04805.pdf)__

__[Growing Neural Networks Achieve Flatter Minima](https://hal.archives-ouvertes.fr/hal-03402267/document)__

## Liens utiles

__[Documentation des transformers HuggingFace](https://huggingface.co/docs/transformers/main/en/index)__\
__[Grid 5000 documentation](https://www.grid5000.fr/w/Getting_Started)__

## Réserver un gpu sur Grid5000 et run le fichier test 
```
ssh alouzali@access.grid5000.fr
ssh nancy
source /home/alouzali/achenv/bin/activate
oarsub -l gpu=1 -I -q production
python finetuning_distilbert.py
```
