## The Core Concept You Should Know About

HNSW organizes vectors into multi-layered graph structure. drawing inspiration from the 'Skip List' data structure but applying it to spatial graphs.

### The Multi Layer Structure

Instead of searching through all vectors sequentially, HSNW builds a hierarchy of graphs:

- **Top Layer**: contains fewer vector with long range links. This allows the search to quickly hop across vast areas of the vector space.
- **Bottom Layers**: Contain increasingly dense cluster of vectors with shorter, localized links for fine-grained accuracy.


### Navigable Small World Routing
The search starts at a global entry point on the top layer. The algorithm performs a greedy search, moving to the closest neighbor in that layer until it reaches a local minimum (a point where no other connected node is closer to the query). 

Once stuck, it drops down to the corresponding node in the next layer below and resumes the search. This process repeats until it reaches **Layer 0**, where it pinpoints the exact closest neighbors.


### The FAISS Advantage

While the native HNSW algorithm is highly accurate, it can be extremely memory-intensive because it stores both the raw high-dimensional vectors and the explicit graph edges in RAM. FAISS addresses these bottlenecks with several key optimizations:


### Memory Compression via Quantization
FAISS allows you to combine HNSW with **Product Quantization (PQ)** or **Scalar Quantization (SQ)**. 
- Instead of storing raw floating-point vectors at the graph nodes, FAISS can compress them into compact byte codes (`IndexHNSW2Level` or `IndexHNSWPQ`).
- This drastically reduces the RAM footprint, allowing you to fit millions of high-dimensional vectors onto a single machine.