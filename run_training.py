import torch
from datasets import load_dataset
from nnsight import LanguageModel

from dictionary_learning import ActivationBuffer, AutoEncoder
from dictionary_learning.trainers import StandardTrainer
from dictionary_learning.training import trainSAE

def main():
    """
    A simple training run for a standard SAE.
    """

    # Model and device configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "EleutherAI/pythia-70m-deduped"

    # Load the model
    print(f"Loading model: {model_name}...")
    model = LanguageModel(
        model_name,
        device_map=device,
    )
    # The submodule whose activations we'll be training our SAE on.
    # We'll use the MLP from layer 1.
    submodule = model.gpt_neox.layers[1].mlp
    # The dimension of the submodule's output. Pythia-70M's MLP has a 2048-dim hidden state
    # but the output is 512 to match the residual stream.
    activation_dim = 512
    # The number of features in our dictionary. A common setting is 32x the activation dimension.
    dictionary_size = 32 * activation_dim

    # Load a dataset from Hugging Face
    print("Loading dataset...")
    dataset = load_dataset("NeelNanda/c4-10k", split="train")
    # We need an iterator that returns strings.
    data = iter(dataset["text"])

    # Set up the activation buffer
    print("Setting up activation buffer...")
    # Ensure n_ctxs is an integer to avoid PyTorch size errors
    n_ctxs = 1000  # roughly how many contexts to keep in the buffer

    buffer = ActivationBuffer(
        data=data,
        model=model,
        submodule=submodule,
        d_submodule=activation_dim,
        n_ctxs=n_ctxs,
        device=device,
    )

    # Configuration for the trainer
    trainer_cfg = {
        "trainer": StandardTrainer,
        "dict_class": AutoEncoder,
        "activation_dim": activation_dim,
        "dict_size": dictionary_size,
        "lr": 3e-4,
        "warmup_steps": 100,
        "steps": 2000, # Number of training steps
        "l1_penalty": 1e-2,
        "layer": 1, # for logging purposes
        "lm_name": model_name, # for logging
        "device": device,
        "sparsity_warmup_steps": 10,
    }

    # Train the sparse autoencoder
    print("Starting training...")
    ae = trainSAE(
        data=buffer,
        trainer_configs=[trainer_cfg],
        steps=2000, # Must match 'steps' in trainer_cfg,
        use_wandb=True,
        wandb_entity="jongharyu-mit",
        wandb_project="sae-bench",
        log_steps=50  # log every 50 steps to W&B
    )
    print("Training complete.")
    print(f"SAE saved to sae_results/0/ae.pt")

if __name__ == "__main__":
    main() 