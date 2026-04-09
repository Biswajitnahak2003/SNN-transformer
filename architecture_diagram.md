# Adaptive Timestep SNN Architecture Diagram
# This file contains the Mermaid diagram markup for the architecture
# To render: Use any Mermaid-compatible viewer or online editor

```mermaid
graph TD
    A[Input MRI<br/>128×128×4] --> B[Spiking U-Net<br/>4-stage Encoder]
    B --> C[Bipolar Linear<br/>Self-Attention<br/>8×8×256]
    C --> D[4-stage Decoder<br/>with Skip Connections]
    D --> E[t=1 Preliminary<br/>Segmentation]

    E --> F[CNN Uncertainty Agent<br/>Entropy + Gradients]
    A --> F

    F --> G[Adaptive Timestep Map<br/>T=1 or T=4 per pixel]

    G --> H[Adaptive Temporal Controller]
    D --> H

    H --> I[Final Segmentation Mask<br/>128×128×4 classes]

    style A fill:#e1f5fe,stroke:#0277bd,stroke-width:2px,color:#0277bd
    style B fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px,color:#7b1fa2
    style C fill:#fff3e0,stroke:#f57c00,stroke-width:2px,color:#f57c00
    style D fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px,color:#7b1fa2
    style E fill:#e8f5e8,stroke:#2e7d32,stroke-width:2px,color:#2e7d32
    style F fill:#fff8e1,stroke:#f9a825,stroke-width:2px,color:#f9a825
    style G fill:#e8f5e8,stroke:#2e7d32,stroke-width:2px,color:#2e7d32
    style H fill:#ffebee,stroke:#c62828,stroke-width:2px,color:#c62828
    style I fill:#e1f5fe,stroke:#0277bd,stroke-width:2px,color:#0277bd
```