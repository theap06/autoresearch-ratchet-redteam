# Research Goal

Minimize validation bits-per-byte (val_bpb) on the FineWeb validation set.

## Approach

You have full access to modify train.py. Try architectural improvements, hyperparameter tuning, optimizer changes, learning rate schedules, and any other modifications that reduce val_bpb. Each experiment trains for 5 minutes.

## Guidelines

- Make one focused change per experiment
- Write clear commit messages describing what you changed and why
- If an experiment regresses val_bpb, analyze why before trying the next idea
- Keep the code clean and well-commented
