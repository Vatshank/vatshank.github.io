---
title: Flash-Attention
subtitle: An exact algorithm for efficient computation of self-attention.
layout: default
date: 2023-12-01
keywords: blogging, writing
published: true
---


<!-- TODOs:

- [x] HBM accesses calculation -- vanilla vs FA -- Theorem 2 -- M and it being theta(Nd)
- [x] <mark> One figure vs Two? (One with outerloop parallelization and one without?). Leaning towards one without the outerloop parallelization. </mark>
- [x] Simplified pseudocode for the fwd pass? Leaning yes.
- [ ] Cleanup
- [ ] Check code
- [ ] Check math
- [x] Figure out how to add pseudocode properly (ignoring for now)

- [x] big oh notation (see if newcommand works? just did manual mathcal replacement for now)
- [x] Flash Attention to FlashAttention?
- [ ] highlight link color? black underline is not obvious and also, open in new tab?
- [x] table at the end
- [ ] intro and subtitle -- see how greg does it
- [x] conclusion
- - [x] summary
- - [x] Why FA is not all that for inference?
- - [x] FA2 and Flash decoding
- - [x] Bwd pass? Leaning no, but maybe point out the parallelization in the vertical dimension?

- [ ] padding sequences vs not? (appendix?)
- [x] gpu section memory hierarchy
- [ ] note that FA will parallelize over batch sizes * n_heads? Also a (batch, heads) for the vanilla attn?
- [x] <mark> appendix softmax overflow correction pseudocode </mark> Skip it for now? Do after push to GH pages.


Others
- [ ] Citations, gregory gunderson style -- how does he cite posts/things that are not papers?
- - [ ]switch to linking to citations instead of hyperlinks?
- [x] Bio
- [ ] Blog page intro -- taking notes etc
- [ ] font - if using Courier, will have to play with the font size (increase? the latex stuff looks much bigger rn) and line height
- [ ] code font - comic code?
- [x] <mark> get it working with GH pages -- domain name, gemfile changes jekyll pages bundle, change meta data from gregory to yours, keywords etc </mark>
- [ ] quotes -- sean lock carrot, lbj 2 points isnt 2 points
 -->


<!-- TODO: quotes at top -->


Now that [FlashAttention-2](https://arxiv.org/abs/2307.08691){:target="_blank"} has been out for a few months and [Flash-Decoding](https://crfm.stanford.edu/2023/10/12/flashdecoding.html){:target="_blank"} is a thing, let's look at the original [FlashAttention](https://arxiv.org/abs/2205.14135){:target="_blank"} algorithm and understand ~~the thing that helps us build things we don't really understand, but like, a lot faster~~ how it efficiently computes self-attention.

## Background
Let’s review the background material real quick.

### Self-Attention
{% katexmm %}

Given matrices $Q, K, V \in {\R}^{N \times d}$ where $N$ is the sequence length and $d$ is the embedding dimension, self-attention calculates the output $O \in \R ^{N \times d}$ as 

$$ O = softmax \left( \frac {QK^T} {\sqrt{d}} \right) V \tag{1} $$

where the $softmax$ is applied row-wise. 

And there we go. That's really all the deep learning background we need to make sense of this.{% sidenote "1" "But if you want to review self-attention or transformers, check out [this post](https://peterbloem.nl/blog/transformers){:target="_blank"}." %} 

Let's assign variables to the intermediate steps so we can refer back to them later in the post -- $ S =  \frac {QK^T} {\sqrt{d}}$ and $P = softmax(S)$
making our output $O = PV$.


In words, we are performing three operations -- a matrix multiplication, followed by a row-wise softmax, followed by another matrix multiplication. In code{% sidenote "ein" "Using *einsum* here because it is *awesome*. If you are unfamiliar, [this](https://ajcr.net/Basic-guide-to-einsum/){:target="_blank"} should help. If you are familiar but not quite comfortable with it yet, I highly recommend working through the code-snippets in this [excellent paper](https://arxiv.org/abs/1911.02150){:target="_blank"} that introduced Multi-Query-Attention." %}, it looks like so --


```python
import torch

# Initialize Q, K, V
N, d = 256, 16
Q, K, V = torch.randn(3, N, d).unbind(0)

# Self-Attention
S = torch.einsum('nd,md->nm', Q, K) / torch.sqrt(torch.tensor(d)) # Scores
P = torch.softmax(S, dim=1) # Probabilities
O = torch.einsum('nn,nd->nd', P, V) # Output
```

Calculating $O$ this way involves -- 
- $\mathcal{O}(N^2d)$ FLOPs -- the $QK^T$ and $PV$ matrix multiplications are $\mathcal{O}(N^2d)$ operations each, and the softmax takes $\mathcal{O}(N^2)$ operations.
- $\mathcal{O}(N^2)$ memory in addition to the inputs/output  -- this is because the intermediate matrices $S, P \in \R^{N \times N}$ take up $\mathcal{O}(N^2)$ storage.


This $N^2$ dependence limits the maximum sequence length we can put through our model. We will soon see that FlashAttention will reduce this memory burden to $\mathcal{O}(N)$ AND be much faster despite continuing to perform $\mathcal{O}(N^2d)$ operations{% sidenote "exact" "There really isn't a way around performing $\mathcal{O}(N^2d)$ operations for \"exact\" attention. There are other techniques, however, that approximate attention and reduce the number of operations at some expense to model quality. See [Performer](https://arxiv.org/abs/2009.14794){:target="_blank"} and [Linear Attention](https://arxiv.org/abs/2006.16236){:target="_blank"} for two such examples."%} by reducing the number of slow accesses to the GPU main memory. But, first, a little detour.

{% endkatexmm %}

---

### (Online) Softmax
{% katexmm %}

Given an input vector $x \in \R^N $, softmax calculates the output $y \in \R^N$ as 

$$ y_i = \frac {e^{x_i}} {\sum_{j=1}^N e^{x_j}} \tag{2} $$


Let's work through a small problem. Given input vectors $s, v \in \R^N$, calculate $y = p^T v \in \R$ where $p = softmax(s)$. That's easy enough -- first calculate $p$ using Equation $2$ and then calculate the dot product $p^T v$. Now let's add a small constraint -- what if we were given the elements of $s$ one at a time? Since we are seeing these elements one-by-one, we can keep a running sum to calculate the softmax denominator as we loop over our vectors and appropriately scale the previous "estimates" of $y$. Concretely, we can run the following routine for $i = 1, 2, ... , N$ iterations -- 

$$ c^{(i)} = c^{(i - 1)} + e^{s_i} $$

$$ y^{(i)} = \frac {c^{(i - 1)} \cdot y^{(i - 1)} + e^{s_i} \cdot v_i} {c^{(i)}} $$
where $c^{(0)} = 0$ and $y^{(0)} = 0$.

The $c^{(i)}$ is the running sum of the softmax denominator, and in each iteration, we use the previous running sum $c^{(i - 1)}$ to scale our previous answer $y^{(i - 1)}$  and add in the new element we just saw in the form of $ e^{s_i} \cdot v_i$. At the end of $N$ iterations, our "estimate" $y^{(N)}$ is the actual $y$ we wanted to calculate. 

Note that we got the output $y = y^{(N)}$ without ever fully materializing the softmax-ed vector $p$ and by only accessing $s$ one element at a time. You might have noticed the resemblance of our toy problem with Equation $1$. To make it more apparent, we could replace $v \in \R^N$ with $V \in \R ^ {N \times d}$ and observe that our update scheme doesn't change much at all -- we just have to apply the update to $d$ entries of the row $V_i$ at a time. This "online" softmax calculation is the bit that lets FlashAttention bring the memory usage down to $\mathcal{O}(N)$ from $\mathcal{O}(N^2)$ because we will never materialize all of $S$ or $P$ in memory and instead work only with "blocks" of those matrices. But more on that later.

In practice, instead of Equation $2$, softmax is calculated like this --

$$ y_i = \frac {e^{x_i - max(x)}} {\sum_{j=1}^N e^{x_j - max(x)}} \tag{3} $$

This is because we don't want [softmax to overflow](https://jaykmody.com/blog/stable-softmax/){:target="_blank"}.

{% endkatexmm %}
--- 

### GPU Stuff
Kind of obvious when it’s spelled out but something I learnt way later than I am willing to admit is --
<!-- TODO:  do in excalidraw?  -->


```
time-taken-to-do-a-thing-on-a-computer = time-spent-on-data-movement + time-spent-on-actual-compute
```


Time spent on data movement includes things like moving your input data from the main memory to the compute unit, saving/loading any intermediate results, and writing your final output back to the main memory.


The data flow in a GPU i.e. the *memory hierarchy* looks something{% sidenote "mem" "See [this](https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html#gpu-arch){:target="_blank"} for a little more background on GPU architecture and [this](https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/#:~:text=all%20the%20SMs.-,A100%20L2%20cache,larger%20than%20V100%20L2%20cache.){:target="_blank"} for the L2 Cache." %} like this-- 

<center> HBM → L2 Cache → SRAM → Compute </center>

HBM (High Bandwidth Memory) refers to the big but slow memory in your GPU. When you use an A100 NVIDIA GPU with a 40 GB memory, that 40GB is the HBM. SRAM is the much faster but much smaller "on-chip" memory, and there is one of these for every Streaming Multiprocessor (SM). SMs are the compute engines of the GPU with the A100 housing 108 SMs. The A100 HBM has a capacity of 40GB with a memory bandwidth of 1.5TB/s whereas the SRAM has a total capacity of 20MB (192 KB per SM) with a bandwidth of 19TB/s.


All of this is to say -- we should try coding things in a way that lets us reduce any unnecessary reads/writes from/to the HBM and reuse the data in SRAM whenever we can. 

## Flash-Attention 

{% katexmm %}    

Alright. So. Finally.

Flash Attention calculates Equation $1$ in $\mathcal{O}(N)$ memory, down from the $\mathcal{O}(N^2)$ memory requirement of the standard implementation. And while there is no getting around performing $\mathcal{O}(N^2d)$ computations for exact attention, it is still up to 3x faster thanks to the reduced number of slow HBM accesses. 

Here is the core idea -- see how that output $O$ is $\R^{N \times d}$ but the intermediate scores ($S$) and attention matrices ($P$) are $\R ^ {N \times N}$? Well, we could use the online softmax update above to calculate the output $O$ without ever fully materializing the attention matrix $P$ in the HBM. We will load the input matrices $Q, K, V$ in chunks to the SRAM, calculate only the required blocks of the score matrix $S$ and manipulate them with the online softmax update (still in SRAM) until a final chunk of the output $O$ is ready to be written out to the HBM.

Let's make a couple of simplifications before we look at the python-esque pseudo code for the forward pass -- we will assume similar row and column block sizes, and also ignore the softmax overflow correction for now.{% sidenote "overflow" "While it is absolutely necessary to do the correction in practice, it does make the pseudocode a tad bit annoying thanks to the additional bookkeeping needed for the max value correction. I have shoved the version with the overflow correction in the [appendix](#appendix) for the more tranquil-minded." %}

Let $B$ be the block size and $n_B = N / B$ be the number of blocks. For $i \in \{1, 2, ... n_B\}$, we will use $B_i$ to denote the $i$-th block, for example, $Q_{B_i}$ would be the $B \times d$ matrix with contiguous rows $Q_{iB}$ ... $Q_{(i + 1)B - 1}$.
*flash_attn* is the function that is responsible for calculating the output $O$ given input matrices $Q$, $K$, $V$ and block-size $B$. It breaks down the problem by partitioning $Q$ into blocks of $Q_{B_i}$s such that the output $O_{B_{i}}$ corresponding to a $Q_{B_{i}}$ is calculated by *flash_attn_inner*.


Here we go{% sidenote "loop" "The order of inner and outer loops here is reversed from that of the algorithm in the [paper](https://arxiv.org/abs/2205.14135){:target="_blank"}. What we have here is similar to [FlashAttention-2](https://arxiv.org/abs/2307.08691){:target="_blank"} and was originally implemented in the [Triton kernel](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/flash_attn_triton_og.py){:target="_blank"}. This lends itself to easy parallelization of the outer loop over the $Q_{B_i}$s." %} -- 

```python
def flash_attn(Q, K, V, B):
    # N is sequence length, d is embedding dimension, B is block-size.
    N, d = Q.shape
    n_B = N / B
    O = zeros(N, d) # Initialize output O as an N x d matrix.
    
    for i in range(n_B): # NOTE: This loop can be parallelized.
        Bi = indices(i * B, (i + 1) * B)
        Q_Bi = load(Q, Bi) # Load a B x d block of Q.
        O_Bi = flash_attn_inner(Q_Bi, K, V, B, d, n_B)
        store(O_Bi, O) # Store the results for the corresponding B x d block of output O.
    return O

def flash_attn_inner(Q_Bi, K, V, B, d, n_B):
    O_Bi, running_sum = zeros(B, d), zeros(B)
    for j in range(n_B): # Given a fixed block of Q, loop over all blocks of K and V.
        Bj = indices(j * B, (j + 1) * B)
        K_Bj, V_Bj = load(K, Bj), load(V, Bj)
        S_ij = Q_Bi @ transpose(K_Bj) # Scores are calculated only for a small subset of the N x N matrix.
        O_Bi, running_sum = online_softmax(O_Bi, S_ij, V_Bj, running_sum)
    return O_Bi

# NOTE: This is without overflow correction.
def online_softmax(O_Bi, S_ij, V_Bj, running_sum):
    new_running_sum = running_sum + S_ij.exp().sum(dim=1)
    O_Bi = O_Bi * running_sum + S_ij.exp() @ V_Bj 
    O_Bi = O_Bi / new_running_sum
    return O_Bi, new_running_sum
```

Attentive{% sidenote "zinger" "Sorry, gotta get the bad jokes out while I still can. This first blog post has been six years in waiting, there might not be another one." %} readers would have noted that the output row $O_i$ depends on $Q$ only through $Q_i$. So we can calculate $O_{B_i}$ just by looking at the corresponding chunk of $Q_{B_i}$ and thus the outer loop over the $Q_{B_i}$s can be parallelized. 

Figure $1$ is a visual illustration of what the *flash_attn_inner* function is doing. Following the notation in the paper, $L$ and $M$ are used to represent the online statistics for the softmax calculation. $L$ is similar to *running_sum*, while $M$ keeps track of the max values for overflow correction (which we have omitted in our pseudocode). 

<div class='figure'>
    <img src="/assets/fa.gif"
         style="width: 100%; display: block; margin: 0 auto;"/>
    <div class='caption'>
        <span class='caption-label'>Figure 1.</span> The dashed blocks represent the "active" blocks in each iteration of the for-loop inside flash_attn_inner.
    </div>
</div>


Let's count the number of times we access the HBM to convince ourselves that these shenanigans are, in fact, helpful --

- *Good ol' Attention* -- Loading in $Q$, $K$, $V$ takes $\mathcal{O}(Nd)$ reads, reading/writing $S$ and $P$ takes $\mathcal{O}(N^2)$ accesses, and writing out $O$ is another $\mathcal{O}(Nd)$ accesses. In total, that's $\mathcal{O}(Nd + N^2)$ HBM accesses.

- *FlashAttention* --  The total number of HBM accesses is $n_B$ times the number of HBM accesses per call to *flash_attn_inner*. Each *flash_attn_inner* call takes $\mathcal{O}(Nd)$ reads for loading $K$ and $V$. Reading $Q_{B_i}$ and writing $O_{B_i}$ mean additional $\mathcal{O}(NB)$ accesses. That adds up to $\mathcal{O}(Nd)$ per call. Assuming SRAM size of $M$, we want to choose a big enough block-size $B$ such that $M = Bd$ and $n_B = N / B = Nd/M$. This gives us a grand total of $\mathcal{O}(N^2d^2/M)$ HBM accesses.

Typically, $d \approx 100$ and $M \approx 100kB$ making $d^2 < M$, and thus FlashAttention has fewer HBM accesses than vanilla self-attention, which makes it faster. Also, the memory requirement is reduced to $\mathcal{O}(N)$ because all we need to save to the HBM are the statistics (*running_sum* above) required to calculate the online softmax (in addition to our inputs and output, of course), whereas vanilla self-attention would have us store the entire $\R^{N \times N}$ attention matrix.

<br />

In summary, FlashAttention is an exact algorithm for efficient computation of self-attention that improves on the standard self-attention implementation in the following ways --  

|       | Standard Attention |  | Flash-Attention |    
| :---  | :---:   | :---: | :---:
| FLOPs  | $\mathcal{O}(N^2d)$  |  |$\mathcal{O}(N^2d)$ |
| Memory| $\mathcal{O}(N^2)$ |  |$\mathcal{O}(N)$ |
| HBM accesses|  $\mathcal{O}(Nd + N^2)$ |  | $\mathcal{O}(N^2d^2/M)$

---

A note on things that are important but didn't get the real estate they deserve -- 

**FlashAttention backward pass** has an analysis similar to the forward pass -- takes $\mathcal{O}(N)$ extra memory and $\mathcal{O}(N^2d^2/M)$ HBM accesses. The $S$ and $P$ matrices aren't stored for the backward pass so as to not blow up the memory and are instead recomputed from $O$ and the softmax statistics. The $O(N^2)$ dropout mask is not stored either and recomputed from the pseudo-random generator state stored from the forward pass.

**FlashAttention-2** [[paper](https://arxiv.org/abs/2307.08691){:target="_blank"}] further optimizes the original FlashAttention algorithm by -- 
- reducing non-matmul FLOPs. This is important because GPUs have much lower throughput (~10x) for non-matmul FLOPs than matmul FLOPs.
- parallelizing across the sequence dimension (we actually covered this with the reversed loop order thingy above).
- better work partitioning that reduces the need for synchronization and shared memory read/writes.

**Flash-Decoding** [[official post](https://crfm.stanford.edu/2023/10/12/flashdecoding.html){:target="_blank"}] is a specialization of the FlashAttention algorithm to auto-regressive inference, where the query sequence length is 1. We were parallelizing the outer loop across blocks $Q_{B_i}$s, but, for inference, since we will only have a single row in $Q$, we will end up under-utilizing the GPU. FlashDecoding solves this issue by dividing the work along the longer key/value sequence dimensions and followed by a reduce operation to get the final output.

**Non-trivial implementation**. 
Here is the deal -- none of this works unless implemented carefully. Multiple operations need to be fused together to avoid unnecessary kernel launch overheads and reads/writes to HBM. As clean and intuitive as the algorithm itself is, writing all that CUDA code to actually get the speedups that the authors did must have been a lot of work. We have side stepped all those gory details here. The [triton kernels](https://github.com/openai/triton/blob/main/python/tutorials/06-fused-attention.py){:target="_blank"} do offer a more approachable way to get your hands dirty with the core FlashAttention algorithm than the [CUDA/CUTLASS/C implementation](https://github.com/Dao-AILab/flash-attention/){:target="_blank"}, but you will lose some of the finer grained control required to implement things like the work partitioning optimization in FlashAttention-2.

<br />

Fin.

---

## Appendix

Pseudocode of the FlashAttention forward pass with the softmax overflow correction -- 

```python
def flash_attn(Q, K, V, B):
    N, d = Q.shape
    n_B = N / B
    O = zeros(N, d)
    
    for i in range(n_B): # NOTE: This loop can be parallelized.
        Bi = indices(i * B, (i + 1) * B)
        Q_Bi = load(Q, Bi)
        O_Bi = flash_attn_inner(Q_Bi, K, V, B, d, n_B)
        store(O_Bi, O)
    return O

def flash_attn_inner(Q_Bi, K, V, B, d, n_B):
    O_Bi, running_sum, running_max = zeros(B, d), zeros(B), -inf(B)
    for j in range(n_B):
        Bj = indices(j * B, (j + 1) * B)
        K_Bj, V_Bj = load(K, Bj), load(V, Bj)
        S_ij = Q_Bi @ transpose(K_Bj)
        O_Bi, running_sum, running_max = online_softmax(O_Bi, S_ij, V_Bj, running_sum, running_max)
    return O_Bi

# With overflow correction.
def online_softmax(O_Bi, S_ij, V_Bj, running_sum, running_max):
    new_running_max = max(running_max, S_ij.max(dim=1)) # Pointwise max.
    new_running_sum = running_sum * (running_max - new_running_max).exp() + (S_ij - new_running_max).exp().sum(dim=1)
    O_Bi = O_Bi * running_sum * (running_max - curr_running_max).exp() + (S_ij - curr_running_max).exp() @ V_Bj
    O_Bi = O_Bi / new_running_sum
    return O_Bi, new_running_sum, new_running_max

```

{% endkatexmm %}
