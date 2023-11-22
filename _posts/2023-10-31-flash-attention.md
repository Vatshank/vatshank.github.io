---
title: Flash Attention
subtitle: Here is some extra detail about the post.
layout: default
date: 2023-10-28
keywords: blogging, writing
published: true
---


<!-- TODO: how to pseudocode? -->

<mark> Make the voice consistent? Talking to the reader -- you/your vs us/ours v/s I/me? Maybe not switch back and forth too often</mark> 


ToDos:

- [ ] HBM accesses calculation -- vanilla vs FA -- Theorem 2 -- M and it being theta(Nd)
- [ ] One figure vs Two? (One with outerloop parallelization and one without?). Leaning towards one without the outerloop parallelization.
- [ ] Simplified pseudocode for the fwd pass? Leaning yes.
- [ ] Bwd pass? Leaning no, but maybe point out the parallelization in the vertical dimension?
- [ ] Why FA is not all that for inference?
- [ ] FA2 and Flash decoding
- [ ] Cleanup
- [ ] Figure out how to add pseudocode properly

Others
- [ ] Footnotes, sidenotes, styling.
- [ ] Bio
- [ ] Blog page intro -- taking notes etc


<!-- TODO: footnotes. Margins? -->

<!-- TODO: quotes at top -->

Now that Flash-Attention-Dos <mark>(link)</mark> has been out for months, figure it’s time to at least make sense of the original one. Our goal is to understand the algorithm and map it to the FA triton kernel found here <mark>[are we doing this? let’s not for now?] (link)</mark> .

## Background

Let’s review the background material real quick.

### Self-Attention

{% katexmm %}    

Given matrices $Q, K, V \in {\R}^{N \times d}$ where $N$ is the sequence length and $d$ is the embedding dimension, self-attention calculates the output $O \in \R ^{N \times d}$ as 

$$ O = softmax (\frac {QK^T} {\sqrt{d}}) V \tag{1} $$

where the $softmax$ is applied row-wise. 

And there we go. That's really all the deep learning background we need for this <mark> point to the intro to transformers post (but if you are not sure what a transformer is or its been a while, go read this ) </mark>. 

Let's assign variables to the intermediate steps so we can refer back to them later in the post -- $ S =  \frac {QK^T} {\sqrt{d}}$ and $P = softmax(S)$
making our output $ O = PV $.


In words, we are performing three operations -- a matrix multiplication, followed by a row-wise softmax, followed by another matrix multiplication. And in code, it looks like so --
```python
import torch

# Initalize Q, K, V
N, d = 256, 16
Q, K, V = torch.randn(3, N, d).unbind(0)

# Attention
S = torch.einsum('nd,nd->nn', Q, K) / torch.sqrt(torch.tensor(d)) # Scores
P = torch.softmax(scores, dim=1) # Probabilities
O = torch.einsum('nm,nd->nd', probs, V) # Output
```

<mark> make a foot note for batch size and multiple heads (add code with it too), if einsum looks alien to you go read this, if einsum looks somewhat familiar but you are not friendly with it, highly recommend reading this paper on MQA and replicating their code snippets line by line. Point to Naom's MQA paper </mark> -- 

As you can see the calculation is O(N^2) in time and space [write out the exact time and space complexity as a function of N and D].
<!-- And as we all know, O(N^2) bad and O(N) good.  -->
This limits the max sequence length you can put through your model, which is not ideal when you are trying to model long range dependencies in your data [rewrite]. [would be nice to show what max sequence length you can put it in through for N^2 vs N for like a 10B model, 32 heads, 24 layers etc. Too annoying to calculate?].


{% endkatexmm %}


### (Online) Softmax
    
<mark> 1) convince yourself the diff between the online softmax vs online softmax * V product </mark>

<mark> 2) start with the vanilla online softmax or directly go to softmax * V product? Leaning towards softmax * V </mark>

{% katexmm %}

Given an input vector $x \in \R^N $, softmax calculates the output $y \in \R^N$ as 

$$ y_i = \frac {e^{x_i}} {\sum_{j=1}^N e^{x_j}} \tag{2} $$


Let's solve a little problem (<mark> kind of like a 1-d version of Equation 1 </mark>). Given an input vector $s \in \R^N$, calculate $y = p^T v \in \R$ where $p = softmax(s)$. That's easy enough -- calculate $p$ first using Equation $2$ and then calculate the dot product $p^T v$. Now let's add a small constraint -- what if we are given the elements of $s$ one at a time? We could do something like this for $i = 1, 2, ... , N$ -- 

<mark> change to pseudo code? </mark>

$$ c^{(i)} = c^{(i - 1)} + e^{s_i} $$

$$ y^{(i)} = \frac {y^{(i - 1)} \times c^{(i - 1)} + v_i \times e^{s_i} } {c^{(i)} + e^ {s_i}} $$
where $c^{(0)} = 0$ and $y^{(0)} = 0$.


Note that we got the output $y$ without ever materializing the entire softmax'd vector $p$ and only accessing $s$ one element at a time. You might have noticed the resemblance of our toy problem with Equation $1$. To make it more apparent, we could replace $v \in \R^N$ with $V \in \R ^ {N \times d}$ and that doesn't change our update scheme at all -- we will then just apply the update to $d$ entries of the row $V_i$ at a time. <mark> rewrite pseudo code with V instead of v </mark>


In practice, softmax is implemented more like this -- <mark> add eqn scaling by max </max> because you wouldn't want your softmax to cause overflow now, would you? <mark> link to Jak Moody's post </mark>

<mark>highlight this important point: </mark>And we care about this because it lets us calculate the output of the attention operation without keeping all the scores (that take up a lot of memory - quantify in O notation?) around just to calculate the $softmax(s)$.

--- 

{% endkatexmm %}



- Code

    - yes? No? No.

---

### GPU Stuff
<mark> fact check things below </mark>

Kind of obvious when it’s spelled out but something I learnt way later than I am willing to admit is --
<mark> do in excalidraw? <mark>
```
time-taken-to-do-a-thing-on-a-computer = time-spent-on-data-movement + time-spent-on-actual-compute
```

Time spent on data movement would include things like moving your input data from the main memory to the compute unit, saving/loading any intermediate results, and writing your final output back to the main memory.



The data flow in a GPU i.e. the *memory hierarchy* looks something like this --  
(<mark> 
caveat that this is simplified
link to Simone's memory diagram and mention his excellent blog/post 
or link here https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html#gpu-arch ?
</mark>
) 

<mark> do below in excalidraw? <mark>

<center> HBM → L2 Cache → SRAM → Compute </center>

<!-- DRAM → L2 Cache → L1/SRAM → Registers  -->
<!-- (<mark> remove the L2 cache above? </mark>) -->

HBM (High Bandwidth Memory) is the big but slow memory for your GPU. When you use an A100 NVIDIA GPU with a 40 GB memory, that 40GB is your HBM. SRAM is the "on-chip" memory, there is one of these per Streaming Multiprocessor (meaning the compute engines, shortened to SM, with the A100 having 108 SMs) and is much faster but much smaller. (For A100) The GPU HBM has a capacity of 40GB and a memory bandwidth of 1.5TB/s whereas the SRAM has a total capacity of 20MB (192 KB per SM) and a bandwidth of 19TB/s.


<mark> add the l2 numbers from below. Add as a ref in the post for this section? </mark>

[here](https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/#:~:text=all%20the%20SMs.-,A100%20L2%20cache,larger%20than%20V100%20L2%20cache.) 

<!-- - Two facts for us to remember — things to the right get progressively smaller in capacity but faster in access time. Maybe add this line in if we decide to keep the L2 and REgisters in the flow?  -->


All of this is to say -- we should try coding things in a way that lets us reduce any unnecessary reads/writes from/to the HBM and reuse the data loaded in SRAM. 

<!-- Therefore, it’s better for us to reuse the loaded data towards the right (SRAM/SMEM?) and therefore reducing unnecessary accesses to the things on the left (HBM) .
- DRAM, SRAM — moving data from DRAM to SRAM is slow (add numbers dram → sram vs sram → compute/register/processor?) -->


<!-- - Computation and memory are not independent. Compute is preceded by moving inputs from memory to perform the compute and followed by moving results out to HBM. -->



## Flash Attention 

{% katexmm %}    

Alright. So. Finally.

<!-- TODO: figure out the actual memory complexity of FA -->
<mark> lol where did "FA runs in O(N) memory" come from? Read it someplace (in the paper or presentation) or did we just straight up hallucinate it? </mark>
<mark> Looks like it is in the paper after all "and uses less memory—linear in sequence length—than standard attention" . Is it because of the values of $M$? They verify this experimentally?</mark>

Flash Attention lets us calculate Equation $1$ in $O(N)$ memory (<mark> umm this is actually $O(N^2d^2 / M)?$ </mark>), down from the $O(N^2)$ (<mark> O(N^2d) actually? </mark>)> memory requirement of the naive implementation. And while there is no getting around performing $O(N^2)$ computations for exact attention, it is still (<mark> how much </mark>) faster thanks to the reduced number of slow HBM accesses. 

Here is the core idea -- see how that output $O$ is $\R^{N \times d}$ but the intermediate score/attention matrices $S$, $P$ <mark> define P, S </mark> are $\R ^ {N \times N}$? Well, we could calculate the output $O$ without ever fully materializing the attention matrix $P$ in the HBM using the online softmax update (<mark> which is what we did for our code snippet above </mark>). We will load in the input matrices $Q, K, V$ in chunks to the SRAM, calculate only the required blocks of the score matrix $S$ and manipulate them with the online softmax update (still in SRAM) until a final chunk of the output $O$ is ready to be written out to the HBM.


<mark> FWD pass pseudo code </mark>
Let's make this concrete and look at the pseudo code for the forward pass (<mark> footnote here that we are reversing the inner and outerloop from the original paper, similar to FA2 and Triton </mark>) --


We will make a couple of simplifications -- assume similar row and column block sizes and we will ignore the softmax overflow correction for now (while it is necessary to do the correction in practice, it does make the pseudocode much harder to read. I have shoved the version with the overflow correction in the appendix). 

<!-- TODO: too many "Let's" to start the sentence? -->

Let $B$ be the block size and $n_B = N / B$ be the number of blocks. For $i \in \{1, 2, ... n_B\}$, we will use $B_i$ to denote the $i$-th block, for example, $Q_{B_i}$ would be the $B \times d$ matrix with contiguous rows $Q_{iB}$ ... $Q_{(i + 1)B - 1}$.



<!-- TODO: do two versions of pseudocode, with and without the overflow correction. Put the latter in appendix since it will be a little messy -->


<mark> writes as a flash_attn and flash_attn_inner function that does the calculation for Q_i, O_i (just like in the new triton implementation) </mark>

```python
def flash_attn(Q, K, V, B):
    N, d = Q.shape
    n_B = N / B
    O = zeros(N, d)
    # NOTE: this loop can be parallelized
    for i in range(n_B):
        Bi = indices(i * B, (i + 1) * B)
        running_sum = zeros(B)
        Q_Bi = load(Q, Bi)
        O_Bi = flash_attn_inner(Q_Bi, K, V, B, d, n_B)
        store(O_Bi, O, Bi) # Store O_Bi in O on the rows Bi TODO: copy the triton syntax here
    return O

def flash_attn_inner(Q_Bi, K, V, B, d, n_B):
    O_Bi = zeros(B, d)
    for j in range(n_B):
        Bj = indices(j * B, (j + 1) * B)
        K_Bj, V_Bj = load(K, Bj), load(V, Bj)
        S_ij = Q_Bi @ transpose(K_Bj)
        O_Bi, running_sum = online_softmax(O_Bi, S_ij, V_j, running_sum)
    return O_Bi

# NOTE: This is without overflow correction
def online_softmax(O_Bi, S_ij, P_ij, running_max, running_exp_sum):
    new_running_sum = running_sum + S_ij.exp().sum(dim=1)
    O_Bi = O_Bi * running_sum + S_ij.exp() @ V_j 
    O_Bi = O_Bi / new_running_sum
    return O_Bi, new_running_sum
```


```python
def flash_attn(Q, K, V, B):
    N, d = Q.shape
    n_B = N / B
    O = zeros(N, d)
    for i in range(n_B): # NOTE: this loop can be parallelized
        Bi = indices(i * B, (i + 1) * B)
        running_sum = zeros(B)
        Q_Bi, O_Bi = load(Q, Bi), load(O, Bi)
        for j in range(n_B):
            K_Bj, V_Bj = load(K, Bj), load(V, Bj)
            S_ij = Q_Bi @ transpose(K_Bj)
            O_Bi, running_sum = online_softmax(O_Bi, S_ij, V_j, running_sum)
        store(O_Bi)
        
    return O



# Without overflow correction
def online_softmax(O_Bi, S_ij, P_ij, running_max, running_exp_sum):
    new_running_sum = running_sum + S_ij.exp().sum(dim=1)
    O_Bi = O_Bi * running_sum + S_ij.exp() @ V_j 
    O_Bi = O_Bi / new_running_sum
    return O_Bi, new_running_sum

```

<mark> shove in the appendix </mark>
```python
# With overflow correction

def online_softmax(O_Bi, S_ij, P_ij, running_max, running_exp_sum):
    curr_max = S_ij.max(dim=1)
    new_running_max = max(running_max, curr_max)

    new_running_sum =  running_exp_sum * exp(running_max - new_running_max) + exp(curr_max - new_running_max).sum(dim=1) 

    O_Bi = O_Bi * running_sum   + V_j * exp(S_ij - new_running_max)
    O_Bi = O_Bi / new_running_sum
    return O_Bi, new_running_max, new_running_sum


```

Let's count <mark> the number of HBM accesses?  </mark> to convince ourselves that these shenanigans do in fact reduce memory accesses to the HBM -- <mark>calculation from Theorem 2 and refer to it in a footnote </mark>
1. Good ol' Attention -- Loading in $Q$, $K$, $V$ is $O(Nd)$, reading/writing $S$ and $P$ is $O (N^2) $, and writing out $O$ is $O(Nd)$. In total, that's $O (Nd + N^2)$.

2. Flash Attention --  Loading in Q, K, V ( $O(Nd)$ ), Writing out O ($O(Nd)$). In total, that's $O (Nd)$

Let's look at this gif that took way too much time to make -- 

<!-- ![Flash Attention](/assets/anim.gif) -->

<img src="/assets/anim.gif" width="500px" height="500px" style="object-fit: contain" />

Attentive (<mark> hehe ^footnote </mark>) readers would have noted that the row $O_i$ depends on $Q$ only through $Q_i$. So you can calculate a chunk of $O$ just by looking at the corresponding chunk of $Q$. (<mark> use this to explain the triton 2d parallelism after explaining the actual core idea (below?) of algorithm? </mark>)

<!-- Would be nice if we could do O(N). Enter Flash Attention — it is an exact attention (side note about approximate attn) will give you O(N) in memory, O(N^2) in time (but actually much faster!) Let’s see how. -->

*(Hmm is O(N) talking about SRAM usage or number of reads/writes? — the device memory is still the same, no?)*

- *I guess FA doesnt ever fully materialize the full N^2 attention matrix and directly calculate the output matrix (which is NxD), which is why FA is O(N) memory?*
- *And it is faster (despite still being O(N^2) for time plus doing additional recomputations (activations?) ) because of the much fewer reads and writes to DRAM? How many reads and writes did we have before? They have that in the theorems? ( — Yes, let’s read theorem 2 and proof)*

[online softmax — write a simplified 1d example] (<mark> put this in the background section? </mark>)

Need the [numerical correction here](https://jaykmody.com/blog/stable-softmax/).


<mark> add a simplified pseudo code for the fwd pass? </mark>

**FWD pass**

Theorem 2 — HBM accesses

- Standard attn
    - (step 1) What is the number of HBM accesses for matmul of two Nxd matrices? How are they getting theta(Nd + N^2)? How is M not showing up in this? Why is d in there?
        - This calculation seems more like data read/written as opposed to number of HBM accesses, no?
    - (step 2)

[Figure showing the block by block computation.]

[]()

[show relevant code snippets?]

[Not as effective for inference. Why?]

[parallelizing along seq dim — FA2 and similar idea in flash decoding]

[Bwd pass — leave as a TODO?]

Prop 3 — can’t have fewer HBM access asymptotically for exact attention.

Recap

- Lowers memory usage by not materializing the full attention matrix and calculating it in blocks.
- Runs faster, not by reducing computations (actually does more compute) by reducing the reads/writes from/to HBM
- Helps training but not inference (yeah?)

{% endkatexmm %}
