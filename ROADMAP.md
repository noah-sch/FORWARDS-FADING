# FSF ROADMAP

> [!NOTE]
> Some tasks may be added to the checklist during the project.

---

## 1. ENVIRONMENT CONFIGURATION

### 1.1 Setup Initial
- [ ] Create a virtual python environment 
- [x] Install required libs (`requirements.txt`)
- [x] Git configuration + repository
- [ ] Checking GPUs availability

### 1.2 Project's structure
- [ ] Create the folder's structure
  ```
  FORWARDS-FADING/
  ├── data/              # Training datas
  ├── models/            # Model's source code
  ├── checkpoints/       # Models checkpoints
  ├── webapp/            # Website / web application
  ├── rag/               # RAG system
  └── docs/              # Documentation
  ```
- [ ] Create a `.gitignore`

---

## 2. LLM DEV

### 2.1 DATA 
- [ ] Build a database
  - [ ] Find sources (Wikipedia, books, articles, etc.)
  - [ ] Clean datas (remove HTML, etc.)
  - [ ] Split in train/validation/test (80/10/10)
- [ ] Data organisation `data/train.txt` & `data/val.txt`
- [ ] Preprocessing script (`scripts/preprocess_data.py`)

### 2.2 Tokenisation
- [ ] Train a personal BOE tokenizer
  - [x] Define vocab size (50k)
  - [ ] Train on the whole dataset
  - [ ] Tokenizer tests
- [ ] Save the final tokenizer (`tokenizer.json`)
- [ ] Create encoder/decoder functions
- [ ] Final validation

### 2.3 Model's architecture
- [ ] Vérifier l'implémentation du Transformer (`transformer_llm.py`)
- [ ] Définir la configuration du modèle
  - [ ] Choose the model's size (small: 100M, medium: 350M, big: 1B+ params)
  - [ ] Define `d_model`, `n_heads`, `n_layers`, etc.
- [ ] Forward pass test
- [ ] verify the text generation

### 2.4 Training
- [ ] Configure hyperparams
  - [ ] Learning rate (3e-4)
  - [ ] Batch size (depends on the GPU)
  - [ ] Warmup steps (1000-5000)
  - [ ] Max steps / epochs
- [ ] Test training
- [ ] Monitoring configuration (Weights & Biases / TensorBoard)
- [ ] Global training
  - [ ] Watch loss
  - [ ] Save checkpoints
  - [ ] Generation test
- [ ] Model's evaluation
- [ ] Select the best checkpoint

### 2.5 Evaluation & Fine-tuning
- [ ] Generation's quality test
  - [ ] Text coherence
  - [ ] Answers diversity
  - [ ] Ability to follow instructions
- [ ] Metrics calculous (perplexity, BLEU, etc.)
- [ ] Fine-tune (if needed)
- [ ] Optimize generation's params (temperature, top_k, top_p)

---

## 3. WEB INTERFACE

### 3.1 API Backend
- [x] Choose framework (FastAPI)
- [ ] Create the REST API
  - [ ] Endpoint `/generate` 
  - [ ] Endpoint `/chat`
  - [ ] Endpoint `/health`
- [ ] Model loading when the webapp is launched 
- [ ] Session chat system
- [ ] Errors + timeouts system
- [ ] Request queue system
- [ ] API tests

### 3.2 Frontend
- [x] Choose the framework (React)
- [ ] Create the homepage
- [ ] Create the chat interface
- [ ] Implement the API communication
- [ ] CSS styling

### 3.3 Documentation page
- [ ] Create an "About FSF" page
  - [ ] Transformers explications
  - [ ] Schemas
  - [ ] Statistics (nb params, etc.)
- [ ] Create a "Technic" page
  - [ ] Stack 
  - [ ] Training process
  - [ ] Model's limits
- [ ] Add usage examples
- [ ] Questions

### 3.4 Optimisation 
- [ ] Performances optimisation
  - [ ] Results in cache
  - [ ] Answers compression
  - [ ] Lazy loading (ressources)
- [ ] Add analytics (optionnal)

---

# Optionnal*

## 4. Système RAG (Retrieval-Augmented Generation)

### 4.1 Préparation de la Base de Connaissances
- [ ] Choisir les documents sources
  - [ ] Documentation technique
  - [ ] Articles de blog
  - [ ] PDFs, livres, etc.
- [ ] Créer un corpus de documents
- [ ] Nettoyer et formater les documents
- [ ] Organiser dans `rag/documents/`

### 4.2 Embeddings et Indexation
- [ ] Choisir un modèle d'embeddings
  - [ ] Sentence-Transformers (recommandé)
  - [ ] OpenAI Embeddings
  - [ ] Modèle custom
- [ ] Installer les dépendances RAG
  ```bash
  pip install sentence-transformers faiss-cpu chromadb langchain
  ```
- [ ] Créer les embeddings des documents
  - [ ] Découper en chunks (200-500 tokens)
  - [ ] Générer les embeddings
  - [ ] Tester la qualité des embeddings
- [ ] Choisir une base vectorielle
  - [ ] FAISS (simple et rapide)
  - [ ] ChromaDB (plus de features)
  - [ ] Pinecone (cloud, payant)
- [ ] Indexer les documents dans la base vectorielle
- [ ] Tester la recherche sémantique

### 4.3 Intégration RAG avec le LLM
- [ ] Créer le pipeline RAG
  ```python
  Question → Recherche → Contexte → LLM → Réponse
  ```
- [ ] Implémenter la recherche de documents pertinents
  - [ ] Top-k retrieval (k=3-5)
  - [ ] Score de similarité
  - [ ] Filtrage des résultats
- [ ] Créer des prompts avec contexte
  ```
  Contexte: [documents trouvés]
  Question: [question utilisateur]
  Réponse:
  ```
- [ ] Intégrer RAG dans l'API
  - [ ] Nouveau endpoint `/rag/chat`
  - [ ] Toggle pour activer/désactiver RAG
- [ ] Tester le système RAG end-to-end
  - [ ] Vérifier que les bonnes sources sont trouvées
  - [ ] Vérifier que le LLM utilise le contexte
  - [ ] Tester avec différents types de questions

### 4.4 Interface RAG
- [ ] Ajouter un toggle RAG dans l'interface chat
- [ ] Afficher les sources utilisées
  - [ ] Titre du document
  - [ ] Extrait pertinent
  - [ ] Score de pertinence
- [ ] Permettre l'upload de documents personnels (optionnel)
- [ ] Créer une page de gestion des documents
  - [ ] Liste des documents indexés
  - [ ] Ajouter/supprimer des documents
  - [ ] Statistiques sur la base de connaissances

### 4.5 Optimisation RAG
- [ ] Améliorer la recherche
  - [ ] Hybrid search (keyword + semantic)
  - [ ] Re-ranking des résultats
  - [ ] Filtres temporels ou par catégorie
- [ ] Optimiser les prompts
  - [ ] Few-shot examples
  - [ ] Instructions claires
  - [ ] Gestion des cas où aucun document n'est pertinent
- [ ] Ajouter la citation des sources
- [ ] Implémenter un cache pour les requêtes fréquentes

---

## ✅ GLOBAL PROGRESSION

```
[░░░░░░░░░░░░░░░░░░░░] 1%

(Approx)
```
