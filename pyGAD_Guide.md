# PyGAD GA Configuration Options (v3.5.0)

## Selection
### Choose the selection operator with parent_selection_type
Supported built-ins (strings you can pass):
* sss – Steady-State Selection.
* rws – Roulette-Wheel Selection.
* sus – Stochastic Universal Sampling.
* rank – Rank Selection.
* random – Uniformly random parents.
* tournament – Tournament Selection (size set by K_tournament).
* For multi-objective problems (fitness function returns a list/tuple/np.array): nsga2 and tournament_nsga2 are recommended; they use non-dominated sorting + crowding distance. 
### Tournament size
* K_tournament=3 by default; increase it to select “stronger” parents. 
### Elitism vs. keeping parents
* keep_elitism (default 1) carries the best K solutions directly into the next generation.
* If keep_elitism > 0, then keep_parents is ignored.
* keep_parents (default -1) controls how many of the selected parents (not offspring) are copied to the next population when keep_elitism=0. (if keep_elitism is not set or >0, keep_parents will be ignored)
* Bounds: 0 ≤ keep_elitism ≤ sol_per_pop, and keep_parents ∈ {0, -1, 1..sol_per_pop}. 
### What the classic selectors mean (quick intuition)
* SSS: 
* Roulette-wheel: probability ∝ fitness.
* SUS: like roulette but uses evenly spaced pointers → lower variance than plain roulette.
* Rank: sort by fitness, assign probabilities by rank (reduces premature takeover).
* Tournament: sample K, pick the best in each mini-contest.
(These are standard GA notions; PyGAD implements them under the names above.) 


## Crossover
### Choose with crossover_type
* single_point – cut once, swap tails.
* two_points – cut twice, swap middle segment.
* uniform – for each gene, pick from parent A or B independently.
* scattered – like uniform but explicitly uses a random binary mask over genes.
* Set crossover_type=None to skip crossover entirely.
* You can also pass a custom function (since 2.16.0) instead of a string. 
### Crossover probability
* crossover_probability: per-parent probability (0.0–1.0) used to decide if that parent participates in mating. If unset, all selected parents are considered. 
### Notes on behavior
* Uniform / scattered pick gene-wise; single/two-point swap contiguous blocks. (General GA definitions.) 

## Mutation
### Choose with mutation_type
* random – add a random value to selected genes, or replace them if mutation_by_replacement=True.
* swap – swap the values of two positions (useful for permutations).
* inversion – reverse the order of a slice.
* scramble – shuffle a slice.
* adaptive – mutation rate depends on whether a solution is above/below the population’s average fitness.
* Set mutation_type=None to skip mutation.
* You can also pass a custom function (since 2.16.0). 
### How many genes mutate? (three interchangeable knobs)
Use one of these; if mutation_probability is set, the other two are ignored.
* mutation_probability: per-gene probability (0.0–1.0).
* mutation_num_genes: an exact count per solution.
* mutation_percent_genes: percent (default string "default" → 10%). 
### Adaptive mutation details
* Set mutation_type="adaptive".
* Provide two rates (low-quality, high-quality) for one of the three knobs above, e.g. mutation_probability=(0.25, 0.1) or mutation_num_genes=(3,1).
* PyGAD classifies a solution as “low-quality” if its fitness < population average. 
### Random mutation value range & mode
* random_mutation_min_val / random_mutation_max_val: range to add (or to sample from when replacing).
* mutation_by_replacement=True: set the gene to a random value instead of adding. (Only affects random mutation.) 
PyGAD
### Interaction with gene_space (very important)
* If a gene’s space is discrete (lists, ranges, or {"low":..,"high":..,"step":..}), PyGAD mutates by replacing the gene with another allowed value.
* If a gene’s space is continuous ({"low":..,"high":..} without step), PyGAD adds a random value, then clips back into the range.
* If a gene’s space is None, init uses init_range_low/high and random-mutation uses random_mutation_min_val/max_val. 
### What swap / inversion / scramble mean (intuition)
* Swap: pick two positions and exchange them.
* Inversion: reverse a contiguous slice.
* Scramble: shuffle the order within a slice. (Standard GA definitions.) 


## Population / initialization
### Ways to provide the starting population
* initial_population: pass a 2-D array you prepared. If set, PyGAD won’t use sol_per_pop or num_genes to create one.
* Otherwise, PyGAD creates the population of size sol_per_pop × num_genes. 
### How genes are typed and initially sampled
* gene_type (default float): set one dtype for all genes, or a list/tuple/np.array to type each gene; can also specify float precision like [float, 2]. 
* init_range_low / init_range_high (defaults −4, +4): random init range when you don’t pass initial_population. 
* gene_space: constrain permissible values per gene. Supports:
    1. single list/range/np.array for a global discrete space,
    2. nested list/tuple for per-gene discrete spaces,
    3. dicts per gene for continuous ({"low":..,"high":..}) or stepped ({"low":..,"high":..,"step":..}) spaces,
    4. None to fall back to init/mutation ranges for that gene. 
### Uniqueness & constraints at init/mutation
* allow_duplicate_genes (default True): if False, PyGAD tries to ensure no two genes in the same chromosome have the same value (given enough unique values).
* gene_constraint: list (length = num_genes) of callables or None. Each callable filters candidate values for the corresponding gene; useful to enforce domain rules (e.g., gene2 > gene1).
* sample_size (default 100): how many candidate values PyGAD samples when it is trying to find a unique or constraint-satisfying value; increase if constraints/uniqueness are hard to satisfy. 
### Reproducibility
* random_seed: sets both Python’s random and NumPy RNGs so runs are repeatable. Leave None for fresh randomness each run. 
### Other useful bits often set alongside init
* num_generations, num_parents_mating, fitness_func (single objective returns a number; multi-objective returns an iterable).
* fitness_batch_size: compute fitness in batches if evaluating is expensive. 

## Quick cheat table (what to pass)
* Selection → parent_selection_type="tournament" (plus K_tournament=5) for robust single-objective; "nsga2" for multi-objective. Control elitism with keep_elitism. 
* Crossover → crossover_type="two_points" or "uniform" for numeric vectors; "scattered" behaves like a mask-based uniform. Use crossover_probability if you need partial mating. 
* Mutation → start with mutation_type="random", set mutation_probability (e.g., 0.1), and choose add vs. replace via mutation_by_replacement. Switch to "adaptive" on tough landscapes. Respect your gene_space. 
* Population/init → either pass initial_population or specify sol_per_pop, num_genes, gene_type, init_range_low/high, gene_space. Add allow_duplicate_genes=False / gene_constraint=[...] if you need structure. 


# References:
- https://pygad.readthedocs.io/en/latest/pygad.html
- https://pygad.readthedocs.io/en/latest/utils.html
- https://www.geeksforgeeks.org/dsa/mutation-algorithms-for-string-manipulation-ga/