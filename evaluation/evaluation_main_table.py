import pandas as pd
from tqdm import tqdm
import os
import concurrent.futures
import multiprocessing
from data_loader import ModelandTask
from method import *

# =========================================
# Configuration Area
# =========================================
MODELS = ["Qwen3-8B-128"]
DATASETS = ["aime24", "aime25", "hmmt25"]
BOOTSTRAP = 1 # Set 1 to disable bootstrapping


# =========================================
# Strategy Configurations
# =========================================
T_groups = [40]
branch_groups = [64]
patiences = [7]
conv = [14]
# kls = [6,8,10,12, 14, 16, 18]
# kss = [2,3,4,5]
warm_ups = [15]
burst_freqs = [1]
branch_configs = [FullReadStrategy()]


## 1D Conv Strategies
for c in conv:  
    branch_configs.append((ConvergenceProbeStrategy(n=c)))


# # Parallel EST v2 Strategies
for T in T_groups:
    for m in branch_groups:
        for prune_patience in patiences:
            for warm_up in warm_ups:
                branch_configs.append(ParallelESTPruning(num_chains=m, T=T, prune_patience=prune_patience, warm_up=warm_up))


# # Parallel EST + only active branch v2 Strategies
for T in T_groups:
    for m in branch_groups:
        for prune_patience in patiences:
            for warm_up in warm_ups:
                branch_configs.append(ParallelESTPruningBurstMajorityOnActiveBranches(num_chains=m, T=T, prune_patience=prune_patience, warm_up=warm_up))

# =========================================
# Solver Configurations
# =========================================
solver_configs = [GreedySolver()]
branch_groups = [64]
for branch in branch_groups:
    solver_configs.append(MajorityVoteSolver(n=branch))


# # ESC Solvers
window_sizes = [8]
branch_groups = [64]
for branch in branch_groups:
    for window in window_sizes:
        if window < branch:
            solver_configs.append(ESCSolver(n=branch, window_size=window))

#ASC Solvers
thresholds = [0.95]
branch_groups = [ 64]
for branch in branch_groups:
    for threshold in thresholds:
        solver_configs.append(ASCSolver(n=branch, threshold=threshold))


# =========================================
# Core Logic
# =========================================

# Global variable for worker processes to cache the task model
worker_task_instance = None

def init_worker(model_name, dataset_name):
    """Initializer for each worker process to load the model once."""
    global worker_task_instance
    print(f"Worker process {os.getpid()} initializing task: {model_name} / {dataset_name}...")
    try:
        worker_task_instance = ModelandTask(model_name, dataset_name)
    except Exception as e:
        print(f"Error initializing worker {os.getpid()}: {e}")

def execute_eval_process(strat_obj, solv_obj):
    """Function executed inside the worker process."""
    global worker_task_instance
    if worker_task_instance is None:
        return None

    try:
        solv_obj.branch_strategy = strat_obj
        
        # Run evaluation using the process-local task instance
        result = worker_task_instance.evaluate(solv_obj, bootstrap_iter=BOOTSTRAP)
        
        entry = {
            "Solver": type(solv_obj).__name__,
            "Strategy": type(strat_obj).__name__,
            "Acc": result['accuracy'],
            "Cost": int(result['avg_cost']),
            "Seq_Cost": int(result.get('avg_sequential_cost', 0)),
            "Std_Acc": result.get('std_accuracy', 0)
        }

        # Adding strategy parameters 
        for k, v in strat_obj.__dict__.items():
            entry[k] = v
        # Adding solver parameters
        for k, v in solv_obj.__dict__.items():
            if k != "branch_strategy":
                entry[f"solver_{k}"] = v
            
        return entry
    except Exception as e:
        # Return error info instead of crashing
        print(f"Error evaluating {type(solv_obj).__name__} with {type(strat_obj).__name__}: {e}")
        return None

def run_matrix_evaluation_multiprocess(model_name, dataset_name):
    # Prepare job arguments (strategies and solvers)
    # Note: We do NOT pass the 'task' object here. 
    jobs = []
    
    for strat_obj in branch_configs:
        for solv_cls in solver_configs:
            solv_name = type(solv_cls).__name__
            strat_name = type(strat_obj).__name__
            if ("Majority" in solv_name or "ASC" in solv_name or "ESC" in solv_name ) and ("Parallel" in strat_name):
                continue
            if ("Greedy" in solv_name) and ("FullRead" in strat_name):
                continue
            if ("ESCSolver" in solv_name or "ASCSolver" in solv_name) and ("Full" not in strat_name):
                continue
            # Arguments for the worker function
            jobs.append((strat_obj, solv_cls))

    print(f"Starting Matrix Eval ({len(jobs)} tasks) with Multiprocessing...")

    # === [Modification Start] ===
    # 临时添加调试模式开关
    DEBUG_MODE = False  # 设置为 True 以启用 breakpoint()

    if DEBUG_MODE:
        print("!!! DEBUG MODE ACTIVE: Running sequentially in main process !!!")
        # 手动初始化全局变量 (模拟 init_worker)
        init_worker(model_name, dataset_name)
        results = []
        for job in tqdm(jobs):
            # 直接在主进程调用函数，此时 breakpoint() 能够生效
            res = execute_eval_process(*job)
            if res:
                results.append(res)
        return results


    num_processes = min(min(32, multiprocessing.cpu_count()),len(jobs)) 
    
    results = []
    
    # Use ProcessPoolExecutor with an initializer to load the model once per process
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=num_processes,
        initializer=init_worker,
        initargs=(model_name, dataset_name)
    ) as executor:
        
        # Submit all jobs
        futures = {executor.submit(execute_eval_process, *job): job for job in jobs}
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(jobs)):
            try:
                res = future.result()
                if res:
                    results.append(res)
            except Exception as exc:
                print(f"Job generated an exception: {exc}")

    return pd.DataFrame(results)

if __name__ == "__main__":
    for model in MODELS:
        for dataset in DATASETS:
            print(f"\nProcessing {model} on {dataset}...")
            # 1. Run evaluation
            try:
                # Use the new multiprocessing function
                data = run_matrix_evaluation_multiprocess(model, dataset)
                
                output_dir = f"matrix_results_{model}"
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                # 4. Save files
                data.to_csv(f"{output_dir}/{dataset}_raw.csv", index=False)
                
                print(f"Saved to {output_dir}/{dataset}_raw.csv")
            except Exception as e:
                print(f"Error processing {model} / {dataset}: {e}")