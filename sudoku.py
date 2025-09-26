"""
4x4 Genetic Sudoku-like Solver (letters) using PyGAD
----------------------------------------------------
- Solves a 4x4 Latin-square-with-2x2-subgrids puzzle with letters.
- Respects given fixed clues.
- Supports multiple initial grids.
- Saves all generations' populations & results to CSV files (one set per case).

Usage:
  - Set LETTERS to any four distinct letters.
  - Define one or more INITIAL_GRIDS with Some letters or None (for blanks).
  - Optionally set TARGET_WORD to e.g. "WORD" to print only edge-word solutions.
  - Run the script.

Author: Chan Man Chak
"""

import os
import csv
import numpy as np
import pygad
from typing import List, Optional, Tuple, Dict, Any
import matplotlib.pyplot as plt

# ----------------------------
# Configuration
# ----------------------------
LETTERS = ['W', 'O', 'R', 'D']     # four distinct symbols; can change
TARGET_WORD: Optional[str] = "WORD" # set None to disable edge filtering

# Multiple initial grids to try; None = blank
INITIAL_GRIDS = [
    # Case A
    [
        ['W', None, None, None],
        [None, 'R', None, None],
        [None, None, 'D', None],
        [None, None, None, 'R'],
    ],
    # Case B (may or may not be solvable)
    [
        [None, None, 'R', None],
        [None, None, None, None],
        ['R', None, None, None],
        [None, 'W', None, None],
    ],
    # Case C (empty)
    [
        [None, None, None, None],
        [None, None, None, None],
        [None, None, None, None],
        [None, None, None, None],
    ],
]

# GA hyperparameters
# How many candidate grids per generation
POP_SIZE = 80
# The GA stops here unless early-stopped first
NUM_GENERATIONS = 1000 #200
# How many parents are chosen to breed each generation
PARENT_MATING = 40
# CROSSOVER options
CROSSOVER_TYPE = "single_point"
# MUTATION options
MUTATION_TYPE = "random"
# Percent of genes (of 16 total) mutated per chromosome.
MUTATION_PERCENT_GENES = 15
# Early stopping patience: once a perfect fitness (1.0) first appears
STOP_IF_PERFECT_AFTER = 15
# For reproducible
RANDOM_SEED = 42

"""
Crossover options:
"single_point" : 1 cut point. Simple, low disruption.
"two_points": 2 cut points. Swaps the middle segment; mixes building blocks a bit more.
"uniform": For each gene, pick from parent A or B with 50% chance. High mixing; can disrupt good substructures.
"scattered": Like uniform but uses a random mask (same idea as uniform; PyGAD treats them similarly).

Mutation options:
"random" : Randomly reassign selected genes from their gene_space. Great with discrete alphabets.
"swap": Pick two gene positions and swap them (helps when permutation structure matters).
"inversion": Pick a segment and reverse it (useful for ordering problems, e.g., TSP).
"scramble": Pick a segment and randomly shuffle its genes (keeps segment but reorders).
"""

# Output directory
OUTPUT_DIR = "ga_csv_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------
# Utilities
# ----------------------------
letter_to_int: Dict[str, int] = {ch: i for i, ch in enumerate(LETTERS)} #{"W":0,"O":1,"R":2,"D":3}
int_to_letter: Dict[int, str] = {i: ch for i, ch in enumerate(LETTERS)} #{0:"W",1:"O",2:"R",3:"D"}

def flatten_grid(grid4x4: List[List[Optional[str]]]) -> List[Optional[int]]:
    """
    Flatten 4x4 grid of letters (or None) into a list of 16 ints (or None).
    Args:
        grid4x4(List[List[Optional[str]]]): 4x4 list of lists with letters or None
        example:   [['W', None, None, None],
                    [None, 'R', None, None],
                    [None, None, 'D', None],
                    [None, None, None, 'R'],]
    Returns:
        out(List[Optional[int]]): List of 16 ints (0..3) or None for blanks
        example:   [0, None, None, None,
                    None, 2, None, None,
                    None, None, 3, None,
                    None, None, None, 2]

    """
    out = []
    for r in range(4):
        for c in range(4):
            v = grid4x4[r][c]
            out.append(None if v is None else letter_to_int[v])
    return out

def to_grid16(chrom: np.ndarray) -> np.ndarray:
    """
    Convert flat chromosome (length 16) to 4x4 numpy int array.
    Args:
        chrom(np.ndarray): 1D numpy array of length 16 with ints (0..3)
        example:   np.array([0, 1, 2, 3,
                             3, 2, 1, 0,
                             1, 0, 3, 2,
                             2, 3, 0, 1])
    Returns:
        (np.ndarray): 4x4 numpy array of ints
        example:   np.array([[0, 1, 2, 3],
                             [3, 2, 1, 0],
                             [1, 0, 3, 2],
                             [2, 3, 0, 1]])
    """
    return np.array(chrom, dtype=int).reshape(4, 4)

def decode_grid(grid_int: np.ndarray) -> List[List[str]]:
    """
    Decode 4x4 numpy int array to 4x4 list of letters.
    Args:
        grid_int(np.ndarray): 4x4 numpy array of ints (0..3)
        example:   np.array([[0, 1, 2, 3],
                             [3, 2, 1, 0],
                             [1, 0, 3, 2],
                             [2, 3, 0, 1]])
    Returns:
        (List[List[str]]): 4x4 list of lists with letters
        example:   [['W', 'O', 'R', 'D'],
                    ['D', 'R', 'O', 'W'],
                    ['O', 'W', 'D', 'R'],
                    ['R', 'D', 'W', 'O']]
    """
    return [[int_to_letter[int(x)] for x in row] for row in grid_int]

def unique_count(seq: List[int]) -> int:
    """
    Count unique elements in a list of ints (0..3).
    Args:
        seq(List[int]): List of ints (0..3)
        example:   [0, 1, 2, 3] or [0, 0, 1, 2] or [1, 1, 1, 1]
    Returns:
        (int): Count of unique elements (0..4)
        example:   4 or 3 or 1
    """
    return len(set(seq))

def subgrid_indices():
    """
    Yields row and column indices for each 2x2 subgrid in a 4x4 grid.
    Args:
        None
    Returns:
        Generator of tuples: (row_indices, col_indices)
    """
    return [
        ([0,1], [0,1]),
        ([0,1], [2,3]),
        ([2,3], [0,1]),
        ([2,3], [2,3]),
    ]

def edges_as_words(grid_letters: List[List[str]]) -> List[str]:
    """
    Extract the four edge words (top, bottom, left, right) from a 4x4 letter grid.
    Args:
        grid_letters(List[List[str]]): 4x4 list of lists with letters
        example:   [['W', 'O', 'R', 'D'],
                    ['D', 'R', 'O', 'W'],
                    ['O', 'W', 'D', 'R'],
                    ['R', 'D', 'W', 'O']]
    Returns:
        (List[str]): List of four strings: [top, bottom, left, right]
        example:   ['WORD', 'RDWO', 'WDOR', 'DROW']
    """
    top = ''.join(grid_letters[0])
    bottom = ''.join(grid_letters[3])
    left = ''.join(row[0] for row in grid_letters)
    right = ''.join(row[3] for row in grid_letters)
    return [top, bottom, left, right]

def grid_to_row_letters(grid_letters: List[List[str]]) -> Dict[str, str]:
    """
    Flatten 4x4 letters into 4 row strings for easy CSV columns.
    Args:
        grid_letters(List[List[str]]): 4x4 list of lists with letters
        example:   [['W', 'O', 'R', 'D'],
                    ['D', 'R', 'O', 'W'],
                    ['O', 'W', 'D', 'R'],
                    ['R', 'D', 'W', 'O']]
    Returns:
        (Dict[str, str]): Dict with keys "row1".."row4" and string values
        example:   {"row1":"WORD", "row2":"DRWO", "row3":"OWDR", "row4":"RDWO"}
    """
    return {
        "row1": ''.join(grid_letters[0]),
        "row2": ''.join(grid_letters[1]),
        "row3": ''.join(grid_letters[2]),
        "row4": ''.join(grid_letters[3]),
    }

def grid16_to_cols_int(grid_int: np.ndarray) -> Dict[str, int]:
    """
    Flatten 4x4 ints into 16 columns c00..c33 for CSV.
    Args:
        grid_int(np.ndarray): 4x4 numpy array of ints (0..3)
        example:   np.array([[0, 1, 2, 3],
                             [3, 2, 1, 0],
                             [1, 0, 3, 2],
                             [2, 3, 0, 1]])
    Returns:
        (Dict[str, int]): Dict with keys "c00".."c33" and int values
        example:   {"c00":0, "c01":1, "c02":2, "c03":3,
                    "c10":3, "c11":2, "c12":1, "c13":0,
                    "c20":1, "c21":0, "c22":3, "c23":2,
                    "c30":2, "c31":3, "c32":0, "c33":1}
    """
    cols = {}
    for r in range(4):
        for c in range(4):
            cols[f"c{r}{c}"] = int(grid_int[r, c])
    return cols

# ----------------------------
# GA Solver with CSV logging
# ----------------------------
class GA4x4SolverCSV:
    def __init__(self, case_id: int, initial_grid: List[List[Optional[str]]], target_word: Optional[str]):
        """
        Initialize the GA solver for a specific case.
        Args:
            case_id(int): Index of the case (for filenames)
            initial_grid(List[List[Optional[str]]]): 4x4 list of lists with letters or None
            target_word(Optional[str]): Target edge word to filter perfect solutions, or None
        Returns:
            None
        """
        self.case_id = case_id # Case index for INITIAL_GRIDS
        self.initial_grid = initial_grid
        self.target_word = target_word

        # Flatted fixed clues as ints (0..3) or None
        self.fixed = flatten_grid(initial_grid)
        # Gene space for PyGAD (list of 16 lists)
        self.gene_space = self._build_gene_space_option(self.fixed)

        # For tracking perfect solutions
        self.perfect_solutions: Dict[Tuple[int, ...], int] = {}  # chrom -> first gen seen
            # perfect_solutions example: {(0,1,2,3,3,2,1,0,1,0,3,2,2,3,0,1): 57}
            # First solution is (0,1,2,3,3,2,1,0,1,0,3,2,2,3,0,1)
            # First perfect solution appeared in generation 57.

        # History of best fitness per generation
        self.best_fitness_history: List[float] = []
            # best_fitness_history example: 
            #self.best_fitness_history = [
            #    0.20,  # best fitness in generation 1
            #    0.25,  # generation 2
            #    0.33,  # generation 3
            #    0.50,  # generation 4
            #    0.75,  # generation 5
            #    0.80,  # generation 6
            #    0.85,  # generation 7
            #    0.95,  # generation 8
            #    1.00,  # generation 9
            #    1.00   # generation 10
            #]

        # For early stopping if perfect fitness persists
        # to track when we first saw a perfect fitness
        self.perfect_seen_at_gen: Optional[int] = None
            # Example: If we first saw a perfect fitness in generation 50,
            # and STOP_IF_PERFECT_AFTER is 15, we will stop if we reach generation 65
            # without losing the perfect fitness.

        # CSV file paths (one set per case)
        self.f_populations = os.path.join(OUTPUT_DIR, f"case_{case_id:02d}__populations.csv")
        self.f_best_hist = os.path.join(OUTPUT_DIR, f"case_{case_id:02d}__best_fitness_history.csv")
        self.f_solutions_all = os.path.join(OUTPUT_DIR, f"case_{case_id:02d}__perfect_solutions.csv")
        self.f_solutions_edge = os.path.join(OUTPUT_DIR, f"case_{case_id:02d}__perfect_solutions_edge_filtered.csv")
        self.f_best_solution = os.path.join(OUTPUT_DIR, f"case_{case_id:02d}__best_solution.csv")
        self.f_initial_grid = os.path.join(OUTPUT_DIR, f"case_{case_id:02d}__initial_grid.csv")

        # Prepare population CSV with header
        with open(self.f_populations, "w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=(
                ["case_id","generation","chromosome_index","fitness"]
                + [f"c{r}{c}" for r in range(4) for c in range(4)]
                + ["row1","row2","row3","row4"]
            ))
            writer.writeheader() # case_id,generation,chromosome_index,fitness,c00,c01,c02,c03,c10,c11,c12,c13,c20,c21,c22,c23,c30,c31,c32,c33,row1,row2,row3,row4

        # Write initial grid (letters, with '.' for blanks)
        with open(self.f_initial_grid, "w", newline="", encoding="utf-8") as fp:
            writer = csv.writer(fp)
            writer.writerow(["Initial Grid (letters; '.' denotes blank)"])
            for r in range(4):
                writer.writerow([self.initial_grid[r][c] if self.initial_grid[r][c] is not None else '.'
                                 for c in range(4)])

    def _build_gene_space_option(self, fixed: List[Optional[int]]) -> List[List[int]]:
        """
        Build gene_space for PyGAD based on fixed clues.
        Args:
        fixed(List[Optional[int]]): List of 16 ints (0..3) or None for blanks
            example:   [0, None, None, None,
                        None, 2, None, None,
                        None, None, 3, None,
                        None, None, None, 2]
        Returns:
        (List[List[int]]): List of 16 lists, each with possible gene values
            example:   [[0], [0,1,2,3], [0,1,2,3], [0,1,2,3],
                        [0,1,2,3], [2], [0,1,2,3], [0,1,2,3],
                        [0,1,2,3], [0,1,2,3], [3], [0,1,2,3],
                        [0,1,2,3], [0,1,2,3], [0,1,2,3], [2]]
        """
        gs = []
        all_vals = list(range(4))
        for v in fixed:
            gs.append([v] if v is not None else all_vals)
        return gs

    def _fitness(self, ga_instance: pygad.GA, solution: np.ndarray, solution_idx: int) -> float:
        """
        Fitness function: 1 / (1 + total_violations)
        Args:
            ga_instance(pygad.GA): The GA instance
            solution(np.ndarray): 1D numpy array of length 16 with ints (0..3)
            solution_idx(int): Index of the solution in the population
        Returns:
            (float): Fitness value (0.0 to 1.0)
        1. Convert solution to 4x4 grid
        2. Count violations:
            - Fixed clues not matching (penalty 10 per mismatch)
            - Rows with duplicates (4 - unique_count)
            - Columns with duplicates (4 - unique_count)
            - 2x2 subgrids with duplicates (4 - unique_count)
        3. Compute fitness = 1 / (1 + violations)
        4. Track perfect solutions (fitness == 1.0)
        5. Return fitness
        """
        grid = to_grid16(solution)
        violations = 0

        # Guard for fixed clues (shouldn't trigger due to gene_space)
        for i, v in enumerate(self.fixed):
            if v is not None and int(solution[i]) != v:
                violations += 10

        # Row/Col checks
        for r in range(4):
            violations += (4 - unique_count(list(grid[r, :])))
        for c in range(4):
            violations += (4 - unique_count(list(grid[:, c])))

        # Subgrids checks
        for rs, cs in subgrid_indices():
            block = [int(grid[r, c]) for r in rs for c in cs]
            violations += (4 - unique_count(block))


        # edge word check (optional)
        #if self.target_word is None:
        #    pass
        #else:
        #    grid_letters = decode_grid(grid)
        #    edge_words = edges_as_words(grid_letters)
        #
        #    if self.target_word not in edge_words:
        #        violations += 4  # penalty if target word not on any edge

        fitness = 1.0 / (1.0 + violations)

        # Track perfects
        if fitness == 1.0:
            key = tuple(int(x) for x in solution)
            if key not in self.perfect_solutions:
                self.perfect_solutions[key] = ga_instance.generations_completed # record which generation first saw this perfect

        return fitness

    def _on_generation(self, ga_instance: pygad.GA):
        """
        Callback after each generation.
        Args:
            ga_instance(pygad.GA): The GA instance
        Returns:
            None
        1. Save entire population with fitness to CSV
        2. Track best fitness for history
        3. Early stop if perfect fitness persists
        4. Return None
        """
        # Save entire population for this generation
        pop = ga_instance.population
        fits = ga_instance.last_generation_fitness  # aligned list

        with open(self.f_populations, "a", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=(
                ["case_id","generation","chromosome_index","fitness"]
                + [f"c{r}{c}" for r in range(4) for c in range(4)]
                + ["row1","row2","row3","row4"]
            ))
            gen_idx = ga_instance.generations_completed # record which generation 
            for i, chrom in enumerate(pop):
                grid_int = to_grid16(chrom)
                grid_let = decode_grid(grid_int)
                row_letters = grid_to_row_letters(grid_let)
                row_data = {
                    "case_id": self.case_id,
                    "generation": gen_idx,
                    "chromosome_index": i + 1,
                    "fitness": float(fits[i]),
                }
                row_data.update(grid16_to_cols_int(grid_int))
                row_data.update(row_letters)
                writer.writerow(row_data)

        # Track best fitness for charting
        best_f = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]
        self.best_fitness_history.append(float(best_f))

        # Early stop if perfect persists
        if best_f == 1.0:
            if self.perfect_seen_at_gen is None:
                self.perfect_seen_at_gen = ga_instance.generations_completed
            elif ga_instance.generations_completed - self.perfect_seen_at_gen >= STOP_IF_PERFECT_AFTER:
                ga_instance.stop_generation = True

    def _collect_perfect_solutions_letters(self) -> List[List[List[str]]]:
        """
        Collect all unique perfect solutions as letter grids.
        Args:
            None
        Returns:
            (List[List[List[str]]]): List of 4x4 letter grids
            example:   [
                        [['W', 'O', 'R', 'D'],
                         ['D', 'R', 'O', 'W'],
                         ['O', 'W', 'D', 'R'],
                         ['R', 'D', 'W', 'O']],
                        ...
                       ]
        """
        sols = []
        for chrom_tuple, _gen in sorted(self.perfect_solutions.items(), key=lambda kv: kv[1]):
            grid = to_grid16(np.array(chrom_tuple))
            sols.append(decode_grid(grid))
        return sols

    def _filter_by_edge_word(self, grids: List[List[List[str]]]) -> List[List[List[str]]]:
        """
        Filter perfect solutions by target edge word (if set).
        Args:
            grids(List[List[List[str]]]): List of 4x4 letter grids
        Returns:
            (List[List[List[str]]]): Filtered list of 4x4 letter grids
            example:   [
                        [['W', 'O', 'R', 'D'],
                         ['D', 'R', 'O', 'W'],
                         ['O', 'W', 'D', 'R'],
                         ['R', 'D', 'W', 'O']],
                        ...
                       ]
        """
        if self.target_word is None:
            return grids
        tw = self.target_word
        return [g for g in grids if any(w == tw for w in edges_as_words(g))]

    def run(self, seed: int = RANDOM_SEED) -> Dict[str, Any]:
        """
        Run the GA solver and save all outputs to CSV files.
        Args:
            seed(int): Random seed for reproducibility
        Returns:
            (Dict[str, Any]): Summary of results and file paths
            example:   {
                        "case_id": 1,
                        "generations_completed": 57,
                        "best_fitness": 1.0,
                        "best_grid_letters": [['W', 'O', 'R', 'D'],
                                                ['D', 'R', 'O', 'W'],
                                                ['O', 'W', 'D', 'R'],
                                                ['R', 'D', 'W', 'O']],
                        "all_perfect_count": 3,
                        "edge_filtered_count": 1,
                        "files": {
                            "populations": "ga_csv_outputs/case_01__populations.csv",
                            "best_fitness_history": "ga_csv_outputs/case_01__best_fitness_history.csv",
                            "perfect_solutions": "ga_csv_outputs/case_01__perfect_solutions.csv",
                            "perfect_solutions_edge_filtered": "ga_csv_outputs/case_01__perfect_solutions_edge_filtered.csv",
                            "best_solution": "ga_csv_outputs/case_01__best_solution.csv",
                            "initial_grid": "ga_csv_outputs/case_01__initial_grid.csv",
                            },
                        "GA_instance": <pygad.GA object at ...>,  # for possible further analysis
                        }
        1. Configure and run PyGAD GA
        2. Save best fitness history to CSV
        3. Save all perfect solutions to CSV
        4. Save edge-filtered perfect solutions to CSV (if target_word set)
        5. Save best solution (even if imperfect) to CSV
        6. Return summary dict
        """
        # Configure and run GA
        ga = pygad.GA(
            num_generations=NUM_GENERATIONS,
            num_parents_mating=PARENT_MATING,
            fitness_func=self._fitness,
            sol_per_pop=POP_SIZE,
            num_genes=16,
            gene_space=self.gene_space,
            parent_selection_type="sss",
            keep_parents=2,
            crossover_type=CROSSOVER_TYPE,
            mutation_type=MUTATION_TYPE,
            mutation_percent_genes=MUTATION_PERCENT_GENES,
            random_seed=seed,
            on_generation=self._on_generation
        )

        ga.run()
        best_sol, best_fit, _ = ga.best_solution(pop_fitness=ga.last_generation_fitness)
        best_grid_letters = decode_grid(to_grid16(best_sol))

        # Save best fitness history
        with open(self.f_best_hist, "w", newline="", encoding="utf-8") as fp:
            writer = csv.writer(fp)
            writer.writerow(["case_id","generation","best_fitness"])
            for i, v in enumerate(self.best_fitness_history, start=1):
                writer.writerow([self.case_id, i, v])

        # Save all perfect solutions (if any)
        all_perfect = self._collect_perfect_solutions_letters()
        if all_perfect:
            with open(self.f_solutions_all, "w", newline="", encoding="utf-8") as fp:
                writer = csv.writer(fp)
                writer.writerow(["case_id","solution_index","row1","row2","row3","row4"])
                for i, g in enumerate(all_perfect, start=1):
                    rows = [''.join(g[0]), ''.join(g[1]), ''.join(g[2]), ''.join(g[3])]
                    writer.writerow([self.case_id, i, *rows])

        # Save edge-filtered perfect solutions (if requested)
        edge_filtered = self._filter_by_edge_word(all_perfect)
        if self.target_word is not None:
            with open(self.f_solutions_edge, "w", newline="", encoding="utf-8") as fp:
                writer = csv.writer(fp)
                writer.writerow(["case_id","target_word","solution_index","row1","row2","row3","row4"])
                if edge_filtered:
                    for i, g in enumerate(edge_filtered, start=1):
                        rows = [''.join(g[0]), ''.join(g[1]), ''.join(g[2]), ''.join(g[3])]
                        writer.writerow([self.case_id, self.target_word, i, *rows])
                else:
                    # still create an empty file with header
                    pass

        # Save the single best solution (even if imperfect)
        with open(self.f_best_solution, "w", newline="", encoding="utf-8") as fp:
            writer = csv.writer(fp)
            writer.writerow(["case_id","generations_completed","best_fitness"])
            writer.writerow([self.case_id, ga.generations_completed, float(best_fit)])
            writer.writerow([])
            writer.writerow(["Best Grid (letters)"])
            for r in best_grid_letters:
                writer.writerow(r)

        return {
            "case_id": self.case_id,
            "generations_completed": ga.generations_completed,
            "best_fitness": float(best_fit),
            "best_grid_letters": best_grid_letters,
            "all_perfect_count": len(all_perfect),
            "edge_filtered_count": len(edge_filtered) if self.target_word is not None else len(all_perfect),
            "files": {
                "populations": self.f_populations,
                "best_fitness_history": self.f_best_hist,
                "perfect_solutions": self.f_solutions_all,
                "perfect_solutions_edge_filtered": self.f_solutions_edge,
                "best_solution": self.f_best_solution,
                "initial_grid": self.f_initial_grid,
            },
            "GA_instance": ga,  # for possible further analysis 
        }

# ----------------------------
# Run all cases
# ----------------------------
def run_all_cases(initial_grids: List[List[List[Optional[str]]]], target_word: Optional[str]):
    """
    Run GA solver on all initial grids and summarize results.
    Args:
        initial_grids(List[List[List[Optional[str]]]]): List of 4x4 grids with letters or None
        target_word(Optional[str]): Target edge word for filtering (or None)
    Returns:
        None
    1. For each initial grid:
        - Create GA4x4SolverCSV instance
        - Run solver
        - Collect results
    2. Optionally write an index CSV summarizing all cases
    3. Return None
    """
    results = []
    for idx, grid in enumerate(initial_grids, start=1):
        solver = GA4x4SolverCSV(case_id=idx, initial_grid=grid, target_word=target_word)
        out = solver.run(seed=RANDOM_SEED)
        results.append(out)

    # Optional: write an index file
    index_path = os.path.join(OUTPUT_DIR, "index__cases_summary.csv")
    with open(index_path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(["case_id","generations_completed","best_fitness",
                         "all_perfect_count","edge_filtered_count"])
        for i, r in enumerate(results, start=1):
            writer.writerow([i, r["generations_completed"], r["best_fitness"],
                             r["all_perfect_count"], r["edge_filtered_count"]])
    
    # Plot curves
    for r in results:
        case_id = r["case_id"]

        # Fitness curve
        hist = r.get("best_fitness_history") or r["GA_instance"].best_solutions_fitness
        fig = plt.figure(figsize=(8, 4))
        plt.plot(hist)
        plt.title(f"Case {case_id:02d} — Fitness per Generation")
        plt.xlabel("Generation")
        plt.ylabel("Best Fitness")
        plt.tight_layout()
        fig.savefig(f"case_{case_id:02d}__fitness.png", dpi=200)
        plt.close(fig)

        # New-solution rate curve
        #plt.figure(figsize=(8, 4))
        #r["GA_instance"].plot_new_solution_rate()
        #plt.tight_layout()
        #plt.savefig(f"case_{case_id:02d}__new_solution_rate.png", dpi=200)
        #plt.close()

if __name__ == "__main__":
    run_all_cases(INITIAL_GRIDS, TARGET_WORD)