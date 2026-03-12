# diffusion_from_scratch
A clean, modular implementation of Denoising Diffusion Probabilistic Models (DDPM) built entirely from scratch using PyTorch. This project demonstrates the step-by-step process of corrupting data with Gaussian noise and training a U-Net to reverse that process to generate new samples. 

Overview:
Diffusion models have revolutionized generative AI. This repository breaks down the complexity of the forward (noising) and backward (denoising) processes.Forward Process: Gradually adds Gaussian noise to an image according to a variance schedule ($\beta$). Reverse Process: A trained U-Net predicts the noise added at a specific timestep $t$ to recover the original signal. 

Technical Stack:
Framework: PyTorch
Calculations: NumPy & Pandas
Architecture: Custom U-Net with Attention Layers 
Environment: Developed using VS Code


Key Features:
Mathematical Implementation: Hand-coded Gaussian noise schedulers (Linear/Cosine). 
Transformer-Based Attention: Integrated attention mechanisms within the U-Net bottlenecks to capture global dependencies. Custom Training Loop: Optimized for stability with Mean Squared Error (MSE) loss between the predicted and actual noise. Sampling Script: Generates high-fidelity images from pure noise using the learned reverse transition.


The model was evaluated using standard deep learning workflows to ensure the generated samples accurately reflect the training distribution. Note: This project highlights my ability to implement Transformers from scratch and manage complex Deep Learning components  beyond just using pre-built libraries.
