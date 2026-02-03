from abc import ABC, abstractmethod
from collections import Counter, deque
import numpy as np
from scipy.stats import beta

class BaseSolver(ABC):
    """
    Pure Interface. 
    It knows nothing about BranchStrategies. 
    It simply defines that a solver must be callable on a question.
    """
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, question) -> str:
        pass
    
    @abstractmethod
    def description(self) -> str:
        pass
# ==========================================
# Dimension 1: Branch Strategy (Strategy for processing a single branch)
# ==========================================

class BranchStrategy(ABC):
    @abstractmethod
    def execute(self, question) -> str:
        """Obtain a single branch's answer from Question, handling specific probe logic."""
        pass

    @abstractmethod
    def description(self) -> str:
        pass

class FullReadStrategy(BranchStrategy):
    """Normal strategy: Read the entire branch directly until the end."""
    def execute(self, question) -> str:
        return question.get_new_branch_final_answer()

    def description(self) -> str:
        return "Full Read"


class ConvergenceProbeStrategy(BranchStrategy):
    """
    Convergence check strategy (single branch):
    - Each trigger runs `probe_more` consecutively `probe_burst` times (unless finished).
    - The trigger's answer is the LAST probed answer in that burst.
    - Stop early if n consecutive trigger answers are identical.
    - Warmup: do not early-stop in the first `warmup_steps` triggers.
    """
    def __init__(self, n=3, warmup_steps=0, probe_burst=1):
        if n < 1:
            raise ValueError("n must be >= 1")
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be >= 0")
        if probe_burst < 1:
            raise ValueError("probe_burst must be >= 1")
        self.n = n
        self.warmup_steps = warmup_steps
        self.probe_burst = probe_burst

    def execute(self, question) -> str:
        try:
            # 1. Start a new branch
            current_ans, index, is_finish = question.probe_new()
        except (ValueError, IndexError):
            raise IndexError("No more branches available")

        # 2. If n<=1 or finished immediately, return directly
        if self.n <= 1 or is_finish:
            return current_ans

        last_ans = current_ans
        streak = 1
        trigger_id = 0  # counts triggers (each trigger may include a burst)

        # 3. Trigger-by-trigger probe (each trigger = burst of probe_more)
        while not is_finish:
            # ---- burst probing ----
            for _ in range(self.probe_burst):
                if is_finish:
                    break
                current_ans, is_finish = question.probe_more(index)
            # -----------------------

            if current_ans == last_ans:
                streak += 1
            else:
                streak = 1
                last_ans = current_ans

            trigger_id += 1

            # Stop early only after warmup
            if trigger_id >= self.warmup_steps and streak >= self.n:
                return current_ans

        return current_ans

    def description(self) -> str:
        return f"Convergence Probe (n={self.n}, warmup_steps={self.warmup_steps}, probe_burst={self.probe_burst})"

# upgrad version: add warmup + burst
class ParallelConvergenceProbeStrategy(BranchStrategy):
    """
    All branches probe in bursts.
    Each trigger: for each unfinished branch, call `probe_more` consecutively `probe_burst` times.
    Then compute step-wise majority on the last answers.
    Stop when n consecutive majority answers are identical.
    Warmup: do not early-stop in the first `warmup_steps` triggers.
    """
    def __init__(self, n=3, init_num_branches=1, warmup_steps=0, probe_burst=1):
        if n < 1:
            raise ValueError("n must be >= 1")
        if init_num_branches < 1:
            raise ValueError("init_num_branches must be >= 1")
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be >= 0")
        if probe_burst < 1:
            raise ValueError("probe_burst must be >= 1")

        self.n = n
        self.init_num_branches = init_num_branches
        self.warmup_steps = warmup_steps
        self.probe_burst = probe_burst

    def cal_majority_answer(self, answers):
        if not answers:
            return None
        counts = Counter(answers)
        return counts.most_common(1)[0][0]

    def is_all_finished(self, branches):
        return all(b["finished"] for b in branches)

    def execute(self, question) -> str:
        branches = []
        try:
            for _ in range(self.init_num_branches):
                ans, idx, is_finish = question.probe_new()
                branches.append({"index": idx, "last_ans": ans, "finished": is_finish})
        except (ValueError, IndexError):
            raise IndexError("No more branches available")

        if self.n <= 1 or self.is_all_finished(branches):
            return self.cal_majority_answer([b["last_ans"] for b in branches])

        last_ans = self.cal_majority_answer([b["last_ans"] for b in branches])
        current_ans = last_ans
        streak = 1

        trigger_id = 0  # counts "triggers" (each trigger runs a burst of probe_more)

        while not self.is_all_finished(branches):
            # ---- burst probing ----
            for _ in range(self.probe_burst):
                for b in branches:
                    if b["finished"]:
                        continue
                    ans, is_finish = question.probe_more(b["index"])
                    b["last_ans"] = ans
                    b["finished"] = is_finish

                if self.is_all_finished(branches):
                    break
            # -----------------------

            current_step_answers = [b["last_ans"] for b in branches]
            current_ans = self.cal_majority_answer(current_step_answers)

            if current_ans == last_ans:
                streak += 1
            else:
                streak = 1
                last_ans = current_ans

            trigger_id += 1

            # early stop only after warmup (warmup is in "trigger" units)
            if trigger_id >= self.warmup_steps and streak >= self.n:
                return current_ans

        return current_ans

    def description(self) -> str:
        return (f"Parallel Convergence Probe (n={self.n}, warmup_steps={self.warmup_steps}, "
                f"probe_burst={self.probe_burst})")


class ParallelWithWarmupBurstAndPrune(BranchStrategy):
    """
    Parallel Convergence + Warmup + Burst + Conservative Pruning

    Pruning idea:
      - A branch is pruned (no longer probed) only if:
          (1) global majority is confident enough (vote_ratio >= prune_vote_ratio)
          (2) and the branch has disagreed with current majority for prune_disagree_steps consecutive triggers
      - Pruned branches do NOT consume more tokens (we stop calling probe_more on them).
      - Voting is computed over ACTIVE (not pruned) branches by default.
    """
    def __init__(
        self,
        n=18,
        init_num_branches=64,
        warmup_steps=10,
        probe_burst=1,

        # pruning knobs
        enable_prune=True,
        prune_disagree_steps=3,      # consecutive disagree triggers to prune a branch
        prune_vote_ratio=0.5,       # only prune when majority is confident
        min_active_branches=3,       # never prune below this number of active branches
        vote_on_pruned=False,        # usually False; True means pruned branches still vote with their last_ans
    ):
        self.n = n
        self.init_num_branches = init_num_branches
        self.warmup_steps = warmup_steps
        self.probe_burst = probe_burst

        self.enable_prune = enable_prune
        self.prune_disagree_steps = prune_disagree_steps
        self.prune_vote_ratio = prune_vote_ratio
        self.min_active_branches = min_active_branches
        self.vote_on_pruned = vote_on_pruned

    def cal_majority_answer(self, answers):
        if not answers:
            return None
        counts = Counter(answers)
        return counts.most_common(1)[0][0]

    def vote_stats(self, answers):
        """return (majority, top1, top2, total)"""
        if not answers:
            return None, 0, 0, 0
        c = Counter(answers)
        common = c.most_common(2)
        maj = common[0][0]
        top1 = common[0][1]
        top2 = common[1][1] if len(common) > 1 else 0
        total = sum(c.values())
        return maj, top1, top2, total

    def is_all_finished_or_pruned(self, branches):
        return all(b["finished"] or b.get("pruned", False) for b in branches)

    def get_vote_answers(self, branches):
        answers = []
        for b in branches:
            if b.get("pruned", False) and not self.vote_on_pruned:
                continue
            answers.append(b["last_ans"])
        return answers

    def num_active(self, branches):
        return sum(1 for b in branches if not b.get("pruned", False))

    def execute(self, question) -> str:
        branches = []
        try:
            for _ in range(self.init_num_branches):
                ans, idx, is_finish = question.probe_new()
                branches.append({
                    "index": idx,
                    "last_ans": ans,
                    "finished": is_finish,
                    "pruned": False,
                    "disagree_streak": 0,  # consecutive triggers disagreeing with current majority
                })
        except (ValueError, IndexError):
            raise IndexError("No more branches available")

        # early return
        maj, top1, top2, total = self.vote_stats(self.get_vote_answers(branches))
        if self.n <= 1 or self.is_all_finished_or_pruned(branches):
            return maj

        last_maj = maj
        streak = 1
        trigger_id = 0
        current_ans = maj

        while not self.is_all_finished_or_pruned(branches):
            # -------- burst probing over ACTIVE branches only --------
            for _ in range(self.probe_burst):
                for b in branches:
                    if b.get("pruned", False) or b["finished"]:
                        continue
                    ans, fin = question.probe_more(b["index"])
                    b["last_ans"] = ans
                    b["finished"] = fin
                if self.is_all_finished_or_pruned(branches):
                    break
            # --------------------------------------------------------

            # vote (default: only active branches)
            vote_answers = self.get_vote_answers(branches)
            maj, top1, top2, total = self.vote_stats(vote_answers)
            current_ans = maj

            # update majority streak
            if maj == last_maj:
                streak += 1
            else:
                streak = 1
                last_maj = maj

            trigger_id += 1

            vote_ratio = (top1 / total) if total > 0 else 0.0

            # -------- pruning (conservative) --------
            if (
                self.enable_prune
                and trigger_id >= self.warmup_steps  # don't prune during warmup
                and maj is not None
                and vote_ratio >= self.prune_vote_ratio
                and self.num_active(branches) > self.min_active_branches
            ):
                # update disagree streak and prune if needed
                for b in branches:
                    if b.get("pruned", False) or b["finished"]:
                        continue
                    if b["last_ans"] != maj:
                        b["disagree_streak"] += 1
                    else:
                        b["disagree_streak"] = 0

                # prune after updating, but keep at least min_active_branches
                for b in branches:
                    if self.num_active(branches) <= self.min_active_branches:
                        break
                    if (not b.get("pruned", False)) and (not b["finished"]) and b["disagree_streak"] >= self.prune_disagree_steps:
                        b["pruned"] = True
            # ----------------------------------------

            # early stop (after warmup)
            if trigger_id >= self.warmup_steps and streak >= self.n:
                return current_ans

        return current_ans

    def description(self) -> str:
        return (f"Parallel Convergence (n={self.n}, warmup_steps={self.warmup_steps}, "
                f"probe_burst={self.probe_burst}, prune={self.enable_prune}, "
                f"prune_disagree_steps={self.prune_disagree_steps}, prune_vote_ratio={self.prune_vote_ratio}, "
                f"min_active_branches={self.min_active_branches}, vote_on_pruned={self.vote_on_pruned})")


class ParallelEST(BranchStrategy):
    """
    Parallel-EST-v2: Fine-grained Early Stopping
    结合了链间共识(Inter)、链内历史稳定性(Intra)和时间步连续性(Temporal)。
    """
    def __init__(self, 
                 num_chains=4,      # 并行链数 n
                 K=14,               # 考察历史的窗口长度
                 T=2,               # 稳定计数阈值 stable_cnt
                 eps_inter=5.0,     # 链间熵阈值 (越小越一致)
                 eps_intra=5.0,     # 链内变异率阈值 (越小越稳定)
                 max_steps=100):     # 最大步数限制 (防止死循环)
        self.num_chains = num_chains
        self.K = K
        self.T = T
        self.eps_inter = eps_inter
        self.eps_intra = eps_intra
        self.max_steps = max_steps

    def _calculate_entropy(self, answers):
        """计算链间结果的熵 (Inter-chain variance)"""
        counts = Counter(answers)
        probs = [count / len(answers) for count in counts.values()]
        return -sum(p * np.log2(p) for p in probs)

    def _calculate_intra_variance(self, history, winner_ans):
        """计算获胜组内部的历史稳定性 (Intra-chain variance)"""
        if not history: return 1.0
        
        # 仅针对给出当前主流答案(winner_ans)的链进行检查
        variances = []
        for h in history:
            if h[-1] == winner_ans:
                # 取最后 K 个历史回答，计算最高频次的占比
                recent = h[-self.K:]
                max_f = Counter(recent).most_common(1)[0][1]
                v_i = 1 - (max_f / len(recent))
                variances.append(v_i)
        
        # 返回平均变异率（也可以用 max）
        return np.mean(variances) if variances else 1.0

    def execute(self, question) -> str:
        # 1. 初始化并行链
        branches = []
        histories = [[] for _ in range(self.num_chains)]
        for i in range(self.num_chains):
            ans, idx, is_finish = question.probe_new()
            branches.append({"index": idx, "finished": is_finish})
            histories[i].append(ans)

        stable_cnt = 0
        prev_winner = None
        step = 0

        # 2. 迭代推进
        while step < self.max_steps:
            current_answers = []
            all_finished = True

            # 并行推进一步
            for i, branch in enumerate(branches):
                if not branch["finished"]:
                    ans, is_finish = question.probe_more(branch["index"])
                    histories[i].append(ans)
                    branch["finished"] = is_finish
                    all_finished = False
                current_answers.append(histories[i][-1])

            # A. 计算当前步的共识答案 a*
            counts = Counter(current_answers)
            winner_ans = counts.most_common(1)[0][0]

            # B. 检查链间一致性 (Inter-chain)
            h_inter = self._calculate_entropy(current_answers)
            inter_ok = (h_inter <= self.eps_inter)

            # C. 检查获胜组的链内稳定性 (Intra-chain)
            # 筛选出当前投给 winner_ans 的链的历史
            winner_histories = [histories[i] for i, ans in enumerate(current_answers) if ans == winner_ans]
            v_intra = self._calculate_intra_variance(winner_histories, winner_ans)
            intra_ok = (v_intra <= self.eps_intra)

            # D. 时间序列稳定性检查 (Temporal)
            if winner_ans == prev_winner and inter_ok and intra_ok:
                stable_cnt += 1
            else:
                stable_cnt = 0
            
            prev_winner = winner_ans

            # 满足早退条件
            if stable_cnt >= self.T:
                return winner_ans

            if all_finished:
                break
            step += 1
        
        return prev_winner

    def description(self) -> str:
        return f"Parallel-EST (n={self.num_chains}, T={self.T})"


class ParallelESTPruning(ParallelEST):
    def __init__(self, 
                 num_chains=4,      
                 K=1000,               
                 T=3,               
                 eps_inter=5,     
                 eps_intra=5,     
                 prune_patience=8,  
                 warm_up=10):        
        super().__init__(num_chains, K, T, eps_inter, eps_intra)
        self.prune_patience = prune_patience
        self.warm_up = warm_up

    def execute(self, question) -> str:
        branches = []
        histories = [[] for _ in range(self.num_chains)]
        # 记录每条链连续偏离共识的次数
        off_track_counts = [0] * self.num_chains
        
        for i in range(self.num_chains):
            ans, idx, is_finish = question.probe_new()
            branches.append({"index": idx, "finished": is_finish})
            # if ans == "ERR":
            #     ans = histories[i][-1] 
            histories[i].append(ans)

        stable_cnt = 0
        prev_winner = None
        step = 0

        while step < self.max_steps:
            current_answers = []
            alive_count = 0

            # --- [Step 1: 并行生成] ---
            for i, branch in enumerate(branches):
                if not branch["finished"]:
                    ans, is_finish = question.probe_more(branch["index"])
                    histories[i].append(ans)
                    branch["finished"] = is_finish
                current_answers.append(histories[i][-1])
                if not branch["finished"]: alive_count += 1

            # --- [Step 2: 共识计算] ---
            counts = Counter(current_answers)
            winner_ans = counts.most_common(1)[0][0]

            # --- [Step 3: 动态剪枝逻辑] ---
            if step >= self.warm_up and alive_count > 1:
                for i, branch in enumerate(branches):
                    if not branch["finished"]:
                        # 如果当前答案不是主流答案
                        if histories[i][-1] != winner_ans:
                            off_track_counts[i] += 1
                        else:
                            off_track_counts[i] = 0
                        
                        # 超过容忍度，直接剪枝
                        if off_track_counts[i] >= self.prune_patience:
                            branch["finished"] = True
                            # print(f"Pruning branch {i} at step {step}")

            # # --- [Step 4: 稳定性评估 (原有逻辑)] ---
            # h_inter = self._calculate_entropy(current_answers)
            # inter_ok = (h_inter <= self.eps_inter)

            # winner_histories = [histories[i] for i, ans in enumerate(current_answers) if ans == winner_ans]
            # v_intra = self._calculate_intra_variance(winner_histories, winner_ans)
            # intra_ok = (v_intra <= self.eps_intra)

            if winner_ans == prev_winner:
                stable_cnt += 1
            else:
                stable_cnt = 0
            
            prev_winner = winner_ans

            # --- [Step 5: 退出判定] ---
            if stable_cnt >= self.T:
                return winner_ans

            # 如果所有链都被剪枝或自然结束
            if all(b["finished"] for b in branches):
                break
            step += 1
        
        return prev_winner

# pruning version with burst probing
class ParallelESTPruningBurst(ParallelEST):
    def __init__(self,
                 num_chains=4,
                 K=5,
                 T=3,
                 eps_inter=5,
                 eps_intra=0.5,
                 prune_patience=5,
                 warm_up=10,
                 probe_burst=1):          # NEW
        super().__init__(num_chains, K, T, eps_inter, eps_intra)
        if probe_burst < 1:
            raise ValueError("probe_burst must be >= 1")
        self.prune_patience = prune_patience
        self.warm_up = warm_up
        self.probe_burst = probe_burst  # NEW

    def execute(self, question) -> str:
        branches = []
        histories = [[] for _ in range(self.num_chains)]
        off_track_counts = [0] * self.num_chains

        for i in range(self.num_chains):
            ans, idx, is_finish = question.probe_new()
            branches.append({"index": idx, "finished": is_finish})
            histories[i].append(ans)

        stable_cnt = 0
        prev_winner = None
        step = 0

        while step < self.max_steps:
            # --- [Step 1: 并行生成] ---
            # burst probing: for each burst, advance every unfinished branch once
            for _ in range(self.probe_burst):
                for i, branch in enumerate(branches):
                    if not branch["finished"]:
                        ans, is_finish = question.probe_more(branch["index"])
                        histories[i].append(ans)
                        branch["finished"] = is_finish
                # 如果 burst 中途所有链都结束了，就提前跳出 burst
                if all(b["finished"] for b in branches):
                    break

            # 注意：共识与剪枝都基于 burst 后的“最新一步答案”
            current_answers = []
            alive_count = 0
            for i, branch in enumerate(branches):
                current_answers.append(histories[i][-1])
                if not branch["finished"]:
                    alive_count += 1

            # --- [Step 2: 共识计算] ---
            counts = Counter(current_answers)
            winner_ans = counts.most_common(1)[0][0]

            # --- [Step 3: 动态剪枝逻辑] ---
            if step >= self.warm_up and alive_count > 1:
                for i, branch in enumerate(branches):
                    if not branch["finished"]:
                        if histories[i][-1] != winner_ans:
                            off_track_counts[i] += 1
                        else:
                            off_track_counts[i] = 0

                        if off_track_counts[i] >= self.prune_patience:
                            branch["finished"] = True

            # --- [Step 4: 稳定性评估 (原有逻辑)] ---
            if winner_ans == prev_winner:
                stable_cnt += 1
            else:
                stable_cnt = 0
            prev_winner = winner_ans

            # --- [Step 5: 退出判定] ---
            if stable_cnt >= self.T:
                return winner_ans

            if all(b["finished"] for b in branches):
                break
            step += 1

        return prev_winner

# upgrading: pruning + burst probing + majority on active chains
class ParallelESTPruningBurstMajorityOnActiveBranches(ParallelEST):
    def __init__(self, 
                 num_chains=64,
                 K=5,
                 T=14,
                 eps_inter=5,
                 eps_intra=5,
                 prune_patience=8,
                 warm_up=10,
                 probe_burst=1):
        super().__init__(num_chains, K, T, eps_inter, eps_intra)
        self.prune_patience = prune_patience
        self.warm_up = warm_up
        self.probe_burst = probe_burst

        if self.probe_burst < 1:
            raise ValueError("probe_burst must be >= 1")

    def execute(self, question) -> str:
        branches = []
        histories = [[] for _ in range(self.num_chains)]
        off_track_counts = [0] * self.num_chains

        # --- init ---
        for i in range(self.num_chains):
            ans, idx, is_finish = question.probe_new()
            branches.append({
                "index": idx,
                "finished": is_finish,  # natural finish (question ends)
                "pruned": False         # NEW: pruned flag
            })
            histories[i].append(ans)

        stable_cnt = 0
        prev_winner = None
        step = 0

        while step < self.max_steps:
            # current_answers_all: 仅用于更新 off_track_counts / histories 对齐
            current_answers_all = [None] * self.num_chains

            active_indices = []
            active_answers = []

            # --- [Step 1: 并行生成 + burst] ---
            for i, branch in enumerate(branches):
                # pruned 的链：不再推进，也不参与 active 集合
                if branch["pruned"]:
                    # 仍然填一下当前答案（用于对齐/调试），但后续投票不使用
                    current_answers_all[i] = histories[i][-1]
                    continue

                # 未 pruned 的链：如果没自然结束，就 burst 推进
                if not branch["finished"]:
                    last_ans = histories[i][-1]
                    for _ in range(self.probe_burst):
                        if branch["finished"]:
                            break
                        ans, is_finish = question.probe_more(branch["index"])
                        branch["finished"] = is_finish
                        last_ans = ans
                    histories[i].append(last_ans)

                # 当前 step 的答案（step-level last）
                cur = histories[i][-1]
                current_answers_all[i] = cur

                # active 定义：not pruned（即使 finished 了，也可以选择参与投票；这里保留参与）
                active_indices.append(i)
                active_answers.append(cur)

            # 如果没有 active（全部 pruned），就退出
            if len(active_indices) == 0:
                break

            # --- [Step 2: 共识计算] --- 只对 active_answers 投票
            counts = Counter(active_answers)
            winner_ans = counts.most_common(1)[0][0]

            # --- [Step 3: 动态剪枝逻辑] --- 只剪 active 且未自然 finished 的链
            if step >= self.warm_up and len(active_indices) > 1:
                for i in active_indices:
                    branch = branches[i]
                    if branch["finished"]:
                        continue  # 自然结束的链不剪（你也可以选择剪）
                    if histories[i][-1] != winner_ans:
                        off_track_counts[i] += 1
                    else:
                        off_track_counts[i] = 0

                    if off_track_counts[i] >= self.prune_patience:
                        branch["pruned"] = True

                # prune 后更新 active 集合（用于后面统计）
                active_indices = [i for i in active_indices if not branches[i]["pruned"]]
                active_answers = [histories[i][-1] for i in active_indices]

                if len(active_indices) == 0:
                    break

                counts = Counter(active_answers)
                winner_ans = counts.most_common(1)[0][0]

            # --- [Step 4: 稳定性评估] --- 只用 active_answers / active winner histories
            
            winner_histories = [
                histories[i]
                for i in active_indices
                if histories[i][-1] == winner_ans
            ]
            

            # 你现在只用 winner 不变作为稳定
            if winner_ans == prev_winner:
                stable_cnt += 1
            else:
                stable_cnt = 0

            prev_winner = winner_ans

            # --- [Step 5: 退出判定] ---
            if stable_cnt >= self.T:
                return winner_ans

            # 如果所有未 pruned 的链都自然 finished，就退出
            if all(b["finished"] or b["pruned"] for b in branches):
                break

            step += 1

        return prev_winner


# ==========================================
# Dimension 2: Solvers
# ==========================================

class StrategyBasedSolver(BaseSolver):
    """
    Intermediate Layer.
    This class implements the logic for solvers that depend on a BranchStrategy
    to fetch samples.
    """
    def __init__(self, branch_strategy=None):
        super().__init__()
        self.branch_strategy = branch_strategy

    def _get_one_sample(self, question):
        """Helper to safely get one sample using the strategy."""
        try:
            return self.branch_strategy.execute(question)
        except (IndexError, ValueError):
            return None
    
    @abstractmethod
    def description(self) -> str:
        pass


# ==========================================
# Concrete Solvers (Inherit from StrategyBasedSolver)
# ==========================================

class GreedySolver(StrategyBasedSolver):
    """Take only the first result."""
    def __call__(self, question) -> str:
        return self._get_one_sample(question) # basically execute the question.

    def description(self) -> str:
        return f"Greedy Solver [Strategy: {self.branch_strategy.description()}]"

class MajorityVoteSolver(StrategyBasedSolver):
    """Fixed N times sampling voting."""
    def __init__(self, branch_strategy=None, n=16):
        super().__init__(branch_strategy)
        self.n = n

    def __call__(self, question) -> str:
        answers = []
        for _ in range(self.n):
            ans = self._get_one_sample(question)
            if ans is not None:
                answers.append(ans)
        
        if not answers:
            return None
        return Counter(answers).most_common(1)[0][0]

    def description(self) -> str:
        return f"Majority Vote (n={self.n}) [Strategy: {self.branch_strategy.description()}]"


class ASCSolver(StrategyBasedSolver):
    """Adaptive-Consistency (ASC)."""
    def __init__(self, branch_strategy=None, n=40, threshold=0.95):
        super().__init__(branch_strategy)
        self.n = n  # 最大采样总数 (Max sampling size k) [cite: 888, 988]
        self.threshold = threshold  # 置信度阈值 (Confidence threshold C_thresh) [cite: 1004, 1069]

    def __call__(self, question):
        all_candidates = [] # 存储观察到的所有样本 (observations) [cite: 975, 991]
        
        # Adaptive-Consistency 采取增量采样策略 
        for n in range(1, self.n + 1):
            # 1. 每次只采样一个样本 [cite: 990, 991]
            ans = self._get_one_sample(question)
            if ans is not None:
                all_candidates.append(ans)
            
            if len(all_candidates) < 2:
                continue
            
            # 2. 统计当前答案分布 [cite: 997, 998]
            counts = Counter(all_candidates)
            # 获取出现次数最多的前两个答案的计数 (v1, v2) [cite: 1027, 1028]
            # 如果目前只有一种唯一答案，则 v2 = 0
            sorted_counts = counts.most_common(2)
            v1 = sorted_counts[0][1]
            v2 = sorted_counts[1][1] if len(sorted_counts) > 1 else 0
            
            # 3. 计算停止准则 (Beta Stopping Criteria) [cite: 1026, 1033, 1485]
            # 计算当前第一名答案 p1 大于第二名答案 p2 的概率 [cite: 1027, 1486]
            # 公式简化为 Beta 分布的累积分布函数 (CDF) [cite: 1028, 1487]
            # 这里的置信度 = 1 - P(p2 >= 0.5) 在参数为 (v2+1, v1+1) 的 Beta 分布下 [cite: 1488]
            confidence = 1 - beta.cdf(0.5, v1 + 1, v2 + 1)
            
            # 4. 如果置信度超过阈值，立即停止采样并上报多数答案 [cite: 1003, 1045]
            if confidence > self.threshold:
                return sorted_counts[0][0] # 直接返回 majority [cite: 979]
        
        # 5. 如果达到最大采样数 k 仍未达到置信度，返回最终的多数投票结果 [cite: 982]
        if not all_candidates:
            return None
            
        return Counter(all_candidates).most_common(1)[0][0]

    def description(self):
        return f"ASC (max={self.n}, th={self.threshold}) [Strategy: {self.branch_strategy.description()}]"


class ESCSolver(StrategyBasedSolver):
    """Early Stopping Consistency (Windowed ESC) - Original ICLR 2024 version without threshold."""
    def __init__(self, branch_strategy=None, n=32, window_size=5):
        super().__init__(branch_strategy)
        self.window_size = window_size  # 窗口大小 (Window size w) [cite: 48, 107]
        self.n = n  # 最大采样总数 (Max sampling size L) [cite: 107]

    def __call__(self, question):
        all_candidates = [] # 存储所有已生成的推理路径 [cite: 107, 115]
        
        # 按照窗口大小分块进行采样 
        # 计算总共需要进行的窗口轮数
        num_windows = self.n // self.window_size
        
        for _ in range(num_windows):
            current_window = []
            
            # 1. 在当前窗口内采样 n 次 [cite: 107]
            for _ in range(self.window_size):
                ans = self._get_one_sample(question)
                if ans is not None:
                    current_window.append(ans)
            
            if not current_window:
                continue
                
            # 将当前窗口的答案加入总候选集 [cite: 107]
            all_candidates.extend(current_window)
            
            # 2. 检查早停准则：窗口内所有答案是否完全一致 (熵为零) 
            # 论文指出这是最严格的阈值，能最大限度维持性能 
            if len(set(current_window)) == 1:
                # 触发早停，直接返回当前窗口的一致答案 [cite: 107, 113]
                return current_window[0]
        
        # 3. 如果所有窗口都采样完仍未触发早停
        # 则对所有采样到的结果进行最终投票 
        if not all_candidates:
            return None
            
        return Counter(all_candidates).most_common(1)[0][0]

    def description(self):
        return f"ESC (win={self.window_size}, max={self.n}) [Strategy: {self.branch_strategy.description()}]"