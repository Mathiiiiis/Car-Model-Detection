import torch

# Assure-toi que ton modèle est déjà en mémoire
model_path = "trained_model.pth"

# Sauvegarde le modèle
torch.save(model.state_dict(), model_path)
print(f"Modèle sauvegardé sous {model_path}")
