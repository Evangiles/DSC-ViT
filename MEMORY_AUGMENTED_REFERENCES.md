# Memory-Augmented Networks: References & Concepts

**Date**: 2025-01-18
**Context**: Exploring memory-based concept learning for DSC-ViT

---

## Motivation

Current clustering approach (D→K→D bottleneck) lacks **cognitive abstraction**:
- ❌ Simple compression, not concept formation
- ❌ No persistent memory of concepts
- ❌ Cannot retrieve "tree" concept when seeing a tree

**Desired behavior**: Human-like categorization
- See "tree" → immediately activate "tree" concept from memory
- Strong categorical perception
- Memory-based abstraction, not just compression

---

## Core References

### 1. Neural Turing Machines (NTM)

**Paper**: "Neural Turing Machines" (Graves et al., 2014, DeepMind)
**ArXiv**: [1410.5401](https://arxiv.org/abs/1410.5401)

**Key Innovation**: Differentiable external memory with read/write operations

**Architecture**:
```python
class NTM:
    def __init__(self, memory_size, memory_dim):
        self.memory = nn.Parameter(torch.randn(memory_size, memory_dim))

    def read(self, query, β=1.0):
        """Content-based addressing"""
        # Similarity to all memory slots
        similarity = F.cosine_similarity(query.unsqueeze(1), self.memory.unsqueeze(0), dim=-1)

        # Softmax with sharpness β
        weights = F.softmax(β * similarity, dim=-1)

        # Weighted read
        read_vector = torch.matmul(weights, self.memory)
        return read_vector, weights

    def write(self, write_weights, erase_vector, add_vector):
        """Erase + Add mechanism"""
        # Erase: multiply by (1 - erase)
        self.memory = self.memory * (1 - torch.outer(write_weights, erase_vector))

        # Add: add new content
        self.memory = self.memory + torch.outer(write_weights, add_vector)
```

**Contributions**:
- First differentiable memory architecture
- Content-based + location-based addressing
- Turing-complete (theoretically)

**Use cases**: Copy tasks, sorting, graph algorithms

---

### 2. Memory-Augmented Neural Networks (MANN)

**Paper**: "Meta-Learning with Memory-Augmented Neural Networks" (Santoro et al., 2016, DeepMind)
**Conference**: ICML 2016

**Key Innovation**: One-shot learning with external memory

**Core Idea**:
```python
# Training phase: See new class, write to memory
for (x, y) in few_shot_examples:
    z = encoder(x)
    memory.write(z, label=y)  # Store concept

# Test phase: Retrieve similar concept from memory
z_query = encoder(x_test)
retrieved, similarity = memory.read(z_query)
prediction = argmax(similarity)  # "This is a tree!"
```

**Memory Update Strategy**:
- **Read**: Content-based (similar to NTM)
- **Write**: Least Recently Used (LRU) replacement
  - New concepts overwrite oldest memory slots
  - Prevents memory overflow

**Application**: Few-shot image classification (Omniglot dataset)

**Connection to our work**:
```python
# Concept storage for segmentation
z_tree = encode(tree_image)
concept_memory.write(z_tree, concept="tree")

# Retrieval during inference
z_query = encode(new_image)
concept, confidence = concept_memory.read(z_query)
# High confidence for "tree" concept!
```

---

### 3. Differentiable Neural Computer (DNC)

**Paper**: "Hybrid computing using a neural network with dynamic external memory" (Graves et al., 2016)
**Journal**: Nature

**Evolution from NTM**:
- **Dynamic memory allocation**: Track which slots are in use
- **Temporal linkage**: Remember order of writes (temporal structure)
- **Usage tracking**: Free unused memory slots

**Architecture**:
```python
class DNC:
    def __init__(self):
        self.memory = []
        self.usage_vector = []      # Which slots are occupied
        self.write_weights = []     # Write attention history
        self.precedence_weights = []  # Temporal order
        self.temporal_links = []    # Forward/backward links

    def allocate_memory(self):
        """Find free memory slot"""
        free_gates = 1 - self.usage_vector
        allocation_weights = free_gates * (1 - precedence)
        return allocation_weights
```

**Key Improvements over NTM**:
- Can handle variable-length sequences
- Better long-term memory retention
- More robust to memory overflow

**Tasks**: Graph tasks, reasoning, program execution

---

### 4. End-to-End Memory Networks

**Paper**: "End-To-End Memory Networks" (Sukhbaatar et al., 2015, Facebook AI)
**ArXiv**: [1503.08895](https://arxiv.org/abs/1503.08895)

**Key Innovation**: Multiple hops over memory (iterative reasoning)

**Architecture**:
```python
class MemoryNetwork:
    def __init__(self, num_hops=3):
        self.memory_keys = []    # For addressing
        self.memory_values = []  # For retrieval
        self.num_hops = num_hops

    def forward(self, query):
        for hop in range(self.num_hops):
            # Attention over memory
            attn = F.softmax(query @ self.memory_keys.T, dim=-1)

            # Read from memory
            retrieved = attn @ self.memory_values

            # Update query for next hop (key insight!)
            query = query + retrieved

        return query
```

**Multi-hop reasoning**:
1. **Hop 1**: Find relevant facts
2. **Hop 2**: Find supporting facts
3. **Hop 3**: Synthesize answer

**Application**: Question answering (bAbI tasks)

**Connection to recursive reasoning**: Multi-hop = iterative refinement (similar to TRM's recursive steps!)

---

### 5. Key-Value Memory Networks

**Paper**: "Key-Value Memory Networks for Directly Reading Documents" (Miller et al., 2016)
**Conference**: EMNLP 2016

**Key Innovation**: Separate keys (for addressing) from values (for content)

**Improvement**:
```python
# Standard Memory Networks
memory = [m1, m2, m3, ...]
attn = softmax(query @ memory.T)
output = attn @ memory  # Same vectors for addressing and content

# Key-Value Memory Networks
memory_keys = [k1, k2, k3, ...]    # For addressing
memory_values = [v1, v2, v3, ...]  # For content
attn = softmax(query @ memory_keys.T)
output = attn @ memory_values  # Flexible content retrieval
```

**Benefit**: Can compress keys while keeping rich values

**Example**:
- Key: "tree" (compact concept ID)
- Value: Detailed tree features (768-dim vector)

---

## Related Work: Prototypical Learning

### Prototypical Networks

**Paper**: "Prototypical Networks for Few-shot Learning" (Snell et al., 2017, NeurIPS)
**ArXiv**: [1703.05175](https://arxiv.org/abs/1703.05175)

**Core Idea**: Each class has a **prototype** (mean of support examples)

```python
class PrototypicalNetwork:
    def compute_prototypes(self, support_set):
        """Compute class prototypes"""
        prototypes = {}
        for class_c in classes:
            examples_c = [x for x in support_set if label(x) == class_c]
            prototypes[class_c] = mean(examples_c)  # Prototype = centroid
        return prototypes

    def classify(self, query, prototypes):
        """Classify by distance to prototypes"""
        distances = {c: euclidean_distance(query, proto)
                     for c, proto in prototypes.items()}
        return argmin(distances)  # Closest prototype wins
```

**Key Insight**: Prototypes = learnable class centroids in embedding space

**Connection to clustering**:
- Soft K-Means: cluster centers = prototypes
- But our current implementation doesn't use them meaningfully!

**Better approach for segmentation**:
```python
class ConceptPrototypes:
    def __init__(self, num_classes=21, latent_dim=768):
        # Learnable prototypes (전형적인 "나무", "개", "자동차")
        self.prototypes = nn.Parameter(torch.randn(num_classes, latent_dim))

    def forward(self, z):
        """Retrieve concept by similarity to prototypes"""
        similarity = F.cosine_similarity(
            z.unsqueeze(1),           # [B, 1, D]
            self.prototypes.unsqueeze(0),  # [1, K, D]
            dim=-1
        )  # [B, K]

        # Softmax → categorical perception
        concept_probs = F.softmax(similarity / temperature, dim=-1)

        # Retrieve prototype (weighted combination)
        concept_embedding = concept_probs @ self.prototypes

        return concept_embedding, concept_probs
```

---

## Synthesis: ConceptMemory for DSC-ViT

### Proposed Architecture

Combining ideas from NTM, MANN, and Prototypical Networks:

```python
class ConceptMemory(nn.Module):
    """
    Memory-augmented concept learning.

    Combines:
    - NTM: Content-based read/write
    - MANN: Concept storage & retrieval
    - Prototypical: Class prototypes as memory slots
    """

    def __init__(self, num_concepts=21, memory_dim=768, memory_size=100):
        super().__init__()

        # Option 1: Fixed prototypes (num_concepts slots)
        self.prototypes = nn.Parameter(torch.randn(num_concepts, memory_dim))

        # Option 2: Dynamic memory (memory_size slots, can store multiple examples)
        self.memory_bank = nn.Parameter(torch.randn(memory_size, memory_dim))
        self.memory_labels = nn.Parameter(torch.zeros(memory_size, num_concepts))

        # Temperature for sharpness
        self.temperature = nn.Parameter(torch.ones(1))

    def read(self, query):
        """
        Content-based retrieval.

        Args:
            query: [B, D] or [B, D, H, W] - reasoning latent
        Returns:
            retrieved: Retrieved concept embedding
            weights: Attention weights (interpretable!)
        """
        # Flatten spatial dimensions if needed
        original_shape = query.shape
        if query.dim() == 4:  # [B, D, H, W]
            B, D, H, W = query.shape
            query = query.permute(0, 2, 3, 1).reshape(-1, D)  # [BHW, D]

        # Compute similarity to all prototypes
        similarity = F.cosine_similarity(
            query.unsqueeze(1),           # [BHW, 1, D]
            self.prototypes.unsqueeze(0),  # [1, K, D]
            dim=-1
        )  # [BHW, K]

        # Softmax with temperature
        weights = F.softmax(similarity / self.temperature, dim=-1)

        # Retrieve weighted prototypes
        retrieved = torch.matmul(weights, self.prototypes)  # [BHW, D]

        # Reshape back to spatial
        if len(original_shape) == 4:
            retrieved = retrieved.reshape(B, H, W, D).permute(0, 3, 1, 2)
            weights = weights.reshape(B, H, W, -1).permute(0, 3, 1, 2)

        return retrieved, weights

    def write(self, content, label, slot_idx=None):
        """
        Write new concept to memory.

        Args:
            content: [D] - concept embedding to store
            label: int - class label
            slot_idx: Optional memory slot (for dynamic memory)
        """
        if slot_idx is None:
            # Update prototype directly
            self.prototypes.data[label] = content
        else:
            # Write to dynamic memory bank
            self.memory_bank.data[slot_idx] = content
            self.memory_labels.data[slot_idx, label] = 1.0

    def update_prototypes_online(self, z, targets, momentum=0.9):
        """
        Online prototype update (like BatchNorm running mean).

        Args:
            z: [B, D, H, W] - current latent features
            targets: [B, H, W] - ground truth labels
            momentum: EMA coefficient
        """
        B, D, H, W = z.shape
        z_flat = z.permute(0, 2, 3, 1).reshape(-1, D)
        targets_flat = targets.reshape(-1)

        # Update prototype for each class
        for c in range(self.prototypes.shape[0]):
            mask = (targets_flat == c)
            if mask.sum() > 0:
                # Mean of features for class c
                class_mean = z_flat[mask].mean(dim=0)

                # EMA update
                self.prototypes.data[c] = (
                    momentum * self.prototypes.data[c] +
                    (1 - momentum) * class_mean
                )
```

### Integration with DSC-ViT

Replace clustering with memory-augmented reasoning:

```python
def update_reasoning_latent_with_memory(x_vit, y, z):
    # Current approach (clustering):
    # z → D→K → soft k-means → K→D → z_new

    # Memory-augmented approach:
    # z → ConceptMemory.read(z) → retrieved_concept → z_new

    # Read from concept memory
    retrieved_concept, concept_weights = concept_memory.read(z)

    # Combine with ViT features and answer
    z_new = retrieved_concept + x_vit + y

    # Optionally: Update memory online
    if training:
        concept_memory.update_prototypes_online(z, targets)

    return z_new, concept_weights  # concept_weights are interpretable!
```

**Benefits**:
1. **Interpretable**: concept_weights show which class prototypes activated
2. **Learnable**: Prototypes adapt to dataset
3. **Memory**: Persistent concept representations
4. **Categorical**: Strong prototype-based perception

---

## Comparison: Clustering vs Memory

| Aspect | Current Clustering | Memory-Augmented |
|--------|-------------------|------------------|
| **Abstraction** | Compression bottleneck | Concept prototypes |
| **Memory** | None (recompute every time) | Persistent prototypes |
| **Interpretability** | Cluster centers (768→21→768) | Prototype activations |
| **Cognitive** | Statistical grouping | Human-like categorization |
| **Retrieval** | None | Content-based addressing |
| **Learning** | Soft K-Means gradients | Prototype updates |

---

## Implementation Roadmap

### Phase 1: Prototype Memory (Simple)
- Replace soft k-means with learnable class prototypes
- Content-based retrieval via cosine similarity
- Supervise with cross-entropy on prototype activations

### Phase 2: Dynamic Memory (Advanced)
- Multiple memory slots per class (store variations)
- LRU replacement for memory updates
- Multi-hop reasoning over memory

### Phase 3: Hierarchical Concepts (Ambitious)
- Tree-structured concept hierarchy
- "Animal" → "Dog" → "Golden Retriever"
- Progressive abstraction across layers

---

## Key Papers to Read

### Must Read (Foundation)
1. **Neural Turing Machines** (Graves et al., 2014) - [arXiv:1410.5401](https://arxiv.org/abs/1410.5401)
2. **Memory-Augmented Neural Networks** (Santoro et al., 2016) - ICML 2016
3. **Prototypical Networks** (Snell et al., 2017) - [arXiv:1703.05175](https://arxiv.org/abs/1703.05175)

### Advanced (Optional)
4. **Differentiable Neural Computer** (Graves et al., 2016) - Nature
5. **End-to-End Memory Networks** (Sukhbaatar et al., 2015) - [arXiv:1503.08895](https://arxiv.org/abs/1503.08895)
6. **Key-Value Memory Networks** (Miller et al., 2016) - EMNLP 2016

### Segmentation Applications
7. **Memory Matching Networks** (Cai et al., 2018) - Few-shot segmentation
8. **Compositional Attention Networks** (Hudson & Manning, 2018) - Visual reasoning

---

## Code References

### Open-Source Implementations
- **PyTorch NTM**: https://github.com/loudinthecloud/pytorch-ntm
- **PyTorch DNC**: https://github.com/ixaxaar/pytorch-dnc
- **Memory Networks (Facebook)**: https://github.com/facebook/MemNN
- **Prototypical Networks**: https://github.com/jakesnell/prototypical-networks

---

## Conclusion

Memory-augmented networks provide a **cognitively plausible** alternative to clustering:
- **Clustering**: Statistical compression (D→K→D bottleneck)
- **Memory**: Concept storage and retrieval (human-like categorization)

Next step: Implement `ConceptMemory` and compare with current clustering approach.

**Hypothesis**: Memory-based reasoning will provide:
1. Better abstraction (meaningful prototypes)
2. Faster learning (fewer parameters, clearer signal)
3. Interpretability (prototype activations)
4. Cognitive validity (human-like memory retrieval)
