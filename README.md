# Neural Transition-Based Dependency Parser

Compact, self-contained implementation of a transition-based dependency parser (PyTorch) for an NLP assignment. The project implements parser transitions, a minibatch parsing driver, and a small feedforward neural model that predicts parser actions from discrete features and pretrained embeddings.

### Key results
- Train UAS: 88.69%
- Test UAS: 89.16%

#### Why this repo
- Educational implementation of a transition-based parser (shift, left-arc, right-arc).
- Small, efficient feedforward model with embeddings, a ReLU hidden layer and dropout.
- Clear separation of parser logic and model code to facilitate experiments and debugging.

### Primary files
- run.py — training / evaluation driver and experiment entrypoint.
- parser_model.py — PyTorch model (ParserModel) and quick sanity checks.
- parser_transitions.py — PartialParse, parse_step, minibatch_parse and unit tests for parser transitions.
- utils/ — dataset loading, vectorization, batching, and helpers (data handling utilities).
- results/ — directory used to save trained weights and run outputs.

### Quickstart (Linux)
1. Create and activate an environment with PyTorch (version >= 1.0 required).
2. Prepare data and embeddings under data/ as expected by utils.
3. Quick parser transition tests:
   - python parser_transitions.py part_c   # tests parse_step / parse
   - python parser_transitions.py part_d   # tests minibatch_parse
4. Train a model (debug mode uses a smaller dataset):
   - python run.py -d      # debug (faster)
   - python run.py         # full training
   Trained weights are saved to results/<timestamp>/model.weights

### Notes and tips
- The model returns raw logits; training uses nn.CrossEntropyLoss (do not apply softmax in the model).
- For reproducible results, keep the saved model.weights files under results/.
- Batch size, learning rate and number of epochs are configurable in run.py (defaults: batch_size=1024, lr=0.0005, n_epochs=10).
- Ensure torch version is >= 1.0.0 (run.py asserts this).