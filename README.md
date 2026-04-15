# **NEURAL STYLE TRANSFER PROJECT - Martiros Saryan**

## **COURSE INFORMATION**

- **UNIVERSITY:** NPUA (National Polytechnic University of Armenia)  
- **COURSE:** Generative Models  
- **LECTURER:** Varazdat Avetisyan  

**STUDENTS:**
- Lilit Azizyan  
- Sona Sargsyan  
- Hrant Nazaryan  

---

## **OVERVIEW**

This project focuses on the implementation and analysis of **Neural Style Transfer (NST)** using two fundamentally different approaches:

1. **Optimization-based method using VGG19**
2. **Diffusion-based method using Stable Diffusion XL + IP-Adapter**

The goal is to transfer artistic style from a style image to a content image while preserving structural information.

All implementations are provided as **Google Colab notebooks**, allowing easy execution and reproducibility.

---

## **MODEL ARCHITECTURES**

### **1. VGG19-BASED NEURAL STYLE TRANSFER**

```text
INPUT  [content_image(3) + style_image(3)]
  │
  ▼
┌─────────────────────────────────────────────────────────┐
│           VGG19 FEATURE EXTRACTION (FROZEN)             │
│                                                         │
│  STYLE PATH:                                            │
│    block1_conv1 ─┐                                      │
│    block2_conv1 ─┼──► Gram Matrix Computation           │
│    block3_conv1 ─┤                                      │
│    block4_conv1 ─┤                                      │
│    block5_conv1 ─┘                                      │
│                                                         │
│  CONTENT PATH:                                          │
│    block5_conv2                                         │
└─────────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────────┐
│                 LOSS COMPUTATION                        │
│                                                         │
│  Style Loss   (weighted Gram matrices)                  │
│  Content Loss (feature reconstruction)                  │
│  TV Loss      (smoothness regularization)               │
│                                                         │
│  Total Loss = Content + Style + TV                      │
└─────────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────────┐
│               OPTIMIZATION PROCESS                      │
│                                                         │
│  Init: 0.7 * content + 0.3 * noise                      │
│  Optimizer: Adam                                        │
│  Backprop on pixels                                     │
└─────────────────────────────────────────────────────────┘
  │
  ▼
OUTPUT  [stylized_image(3)]
```
---

### **2. DIFFUSION-BASED STYLE TRANSFER (SDXL + IP-ADAPTER)**

```text
INPUT  [content_image(3) + style_image(3) + prompt]
  │
  ▼
┌─────────────────────────────────────────────────────────┐
│               TEXT ENCODER (CLIP)                       │
│        prompt → text embeddings                         │
└─────────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────────┐
│               VAE ENCODER                               │
│        content_image → latent space                     │
└─────────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────────┐
│               IP-ADAPTER                                │
│        style_image → style embeddings                   │
│        inject style into diffusion process              │
└─────────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────────┐
│               UNET DENOISING PROCESS                    │
│                                                         │
│  noise → x₁ → x₂ → ... → x_T                            │
│                                                         │
│  conditioned on:                                        │
│    • text embeddings                                    │
│    • style embeddings                                   │
│                                                         │
│  parameters:                                            │
│    strength                                             │
│    guidance_scale                                       │
│    num_inference_steps                                  │
└─────────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────────┐
│               VAE DECODER                               │
│        latent → RGB image                               │
└─────────────────────────────────────────────────────────┘
  │
  ▼
OUTPUT  [stylized_image(3)]
```

---

## **MODEL COMPONENTS & HYPERPARAMETERS**

### **VGG19 NST**

- Image size: 512  
- Content weight: 3e3  
- Style weight: 0.8  
- TV weight: 3.0  
- Learning rate: 0.01  
- Epochs: 30  
- Steps per epoch: 100  

**Layers:**
- Content: block5_conv2  
- Style: block1_conv1 → block5_conv1  

---

### **DIFFUSION MODEL**

- Model: Stable Diffusion XL  
- Adapter: IP-Adapter  

**Hyperparameters:**
- Strength: 0.35 – 0.75  
- Guidance scale: 6.0 – 8.0  
- Steps: 30 – 50  
- Seed: 42  

---

## **LOSS FUNCTIONS*

```math
L_{content} = \| F_{generated} - F_{content} \|^2
```

```math
L_{style} = \sum_{l} w_l \cdot \| G_l^{generated} - G_l^{style} \|^2
```

```math
L_{tv} = \sum_{i,j} \left( |x_{i,j} - x_{i+1,j}| + |x_{i,j} - x_{i,j+1}| \right)
```

```math
L_{total} = \alpha L_{content} + \beta L_{style} + \gamma L_{tv}
```

---

## **HOW TO USE THE CODE**

1. Open the Colab notebook  
2. Mount Google Drive  
3. Set paths for:
   - Content image  
   - Style image  
4. Choose model (VGG19 or Diffusion)  
5. Adjust hyperparameters  
6. Run all cells  
7. Output will be saved to Drive  

---

## **REFERENCES**

1. Sanakoyeu, A., Kotovenko, D., Lang, S., & Ommer, B.  
   *A Style-Aware Content Loss for Real-Time HD Style Transfer.*  
   ECCV, 2018  

2. Singh, A., Jaiswal, V., & Joshi, G.  
   *Neural Style Transfer: A Critical Review.*  
   IEEE Access, 2021  

3. Cao, J.  
   *Neural Style Transfer: A Review and Analysis.*  
   Highlights in Science, Engineering and Technology, 2025  

---

## **NOTES**

- All code is implemented in Google Colab  
- GPU is recommended (T4 or higher)  
- The project is fully reproducible  
