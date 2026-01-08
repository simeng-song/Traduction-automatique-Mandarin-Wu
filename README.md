# Traduction-automatique-Mandarin-Wu

## Présentation du projet
Ce dépôt contient le projet de groupe réalisé dans le cadre du cours **Apprentissage, réseaux de neurones profonds**.
L’objectif du projet est de construire un système de traduction automatique neuronale du mandarin vers le wu (shanghaïen), une langue régionale peu dotée, à l’aide de modèles entraînés à partir de zéro avec Keras.

Ce travail vise à mettre en pratique les concepts fondamentaux des réseaux de neurones profonds appliqués au traitement automatique des langues (TAL), en particulier dans le contexte de la traduction automatique.

## Modèles : Transformer
1. Nombre de couches :
- encodeur : 4 couches
- decodeur : 4 couches
- multi-head self-attentions
2. Type d'encodage  
Tokenisation caractères
3. Type d'apprentissage  
Apprentissage supervisé

#### Baseline

Encodeur (Mandarin) -> Decodeur (Wu) -> Output Layer

#### Hyperparamètres :

| Hyperparamètre | Valeur | Description |
|----------------|--------|-------------|
| `D_MODEL` | 128 | Dimension des embeddings |
| `N_ENC` | 4 | Nombre de couches d'encodeur |
| `N_DEC` | 4 | Nombre de couches de décodeur |
| `N_HEADS` | 8 | Nombre de têtes d'attention |
| `DFF` | 512 | Dimension du feed-forward |
| `DROP` | 0.1 | Taux de dropout |
| `MAX_SRC_LEN` | 50 | Longueur max source |
| `MAX_TGT_LEN` | 50 | Longueur max cible |
| `MAX_VOCAB_SIZE` | 4000 | Taille max du vocabulaire |

#### Callbacks

1. **ModelCheckpoint** :
   - Sauvegarde le meilleur modèle selon `val_loss`
   - Permet de récupérer le modèle optimal

2. **EarlyStopping** :
   - `patience=5` : arrête si pas d'amélioration pendant 5 epochs
   - `restore_best_weights=True` : restaure les meilleurs poids
   - Évite le sur-apprentissage



#### Résultats

| Mandarin | Wu | Evaluation |
|----------|----|------------|
|你好 | 侬好叫吾老 |
|你今天吃饭了吗 | 侬今朝吃饭了伐 |
|我不会说上海话 | 吾伐会的讲呢 |
|今天天气真好 | 今朝天气真呃好 |
|请你帮我一下 | 请侬帮吾一下 |
|祝您开心 | 祝侬开心心 |
