# Growing-Neural-Networks---Stage2A
Ce repository Git contient les codes utilisés lors du stage et qui étaient conscis pour être exécutés en Grid5000 

## Papiers de recherche sur les transformers, réseaux de neurones grossissants...

__[Attention is all you need](https://arxiv.org/pdf/1706.03762.pdf)__

__[BERT](https://arxiv.org/pdf/1810.04805.pdf)__

__[Growing Neural Networks Achieve Flatter Minima](https://hal.archives-ouvertes.fr/hal-03402267/document)__

## Liens utiles

__[Documentation des transformers HuggingFace](https://huggingface.co/docs/transformers/main/en/index)__\
__[Grid 5000 documentation](https://www.grid5000.fr/w/Getting_Started)__  
__[Guide pour l'utilisation de OAR](https://gricad-doc.univ-grenoble-alpes.fr/hpc/joblaunch/)__
## Réserver un gpu sur Grid5000 et run le fichier test en mode interactif
```
ssh alouzali@access.grid5000.fr
ssh nancy
source /home/alouzali/achenv/bin/activate
oarsub -I -l gpu=1 -t exotic
python finetuning_distilbert.py
```
## Réserver un gpu sur Grid5000 et run le fichier test
* Créer le fichier finetuning_distilbert.sh qui exécutera finetuning_distilbert.py plus tard  
`* Contenu du fichier finetuning_distilbert.sh`
 ```
#!/bin/bash
#OAR -n test_script
#OAR -t exotic
#OAR -l /gpu=1
#OAR --stdout result_finetuning.txt    ' Renvoie le résultat sous forme d'un fichier result_finetuning.txt'
python finetuning_distilbert.py
```
* Exécuter finetuning_distilbert.sh via ces deux commandes
```
chmod +x finetuning_distilbert.sh
oarsub -S ./finetuning_distilbert.sh
```
