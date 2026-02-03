import json
from method import BaseSolver
import random
import numpy as np
class Branch:
    def __init__(self, probe_list, branch_tokens, final_answer):
        self.probe_list = probe_list
        self.branch_tokens = branch_tokens
        self.final_answer = final_answer
        self.__cost = 0
        self.__index = 0
    
    def explore(self,probe_freq=500):
        if self.__index < len(self.probe_list):
            answer=self.probe_list[self.__index]
            self.__index += 1
            self.__cost += probe_freq
            return answer,probe_freq,False
        else:
            return self.final_answer, max(0,self.branch_tokens-self.__cost),True
        
    
class Question:
    def __init__(self, infos, seed=42):
        self.__question = infos['question']
        self.__final_answers_trace = infos['final_answers_trace']
        self.__each_branch = [Branch(*branch) for branch in infos['each_branch']]
        self.rng = random.Random(seed)
        self.rng.shuffle(self.__each_branch)
        self.__gold_answer = infos['gold_answer']
        self.probe_freq = infos['probe_freq']
        self.__cost = 0
        self._cost_sequential = []   
        self._cost_per_branch = [0 for _ in range(len(self.__each_branch))]     
        self.__index = 0

    def get_new_branch_final_answer(self):
        branch = self.__each_branch[self.__index]
        self.__index += 1
        self.__cost += branch.branch_tokens
        self._cost_per_branch[self.__index - 1] += branch.branch_tokens
        return branch.final_answer
    
    def probe_new(self):

        if self.__index < len(self.__each_branch):
            branch = self.__each_branch[self.__index]
            branch_answer, cost, isFinish = branch.explore(self.probe_freq)
            self.__cost += cost
            self._cost_per_branch[self.__index] += cost
            self.__index += 1
            return branch_answer,self.__index-1, isFinish
        else:
            raise ValueError("Index out of range for branches.")

    def probe_more(self,index):
        if index<=self.__index:
            branch = self.__each_branch[index]
            branch_answer, cost, isFinish = branch.explore(self.probe_freq)
            self.__cost += cost
            self._cost_per_branch[index] += cost
            return branch_answer, isFinish
        else:
            raise ValueError("Index out of range for branches.")

    def get_cost_per_branch_active(self):
        return list(self._cost_per_branch[:self.__index])

    def get_seq_and_total_tokens(self):
        # 只统计已经启动过的分支：0..self.__index-1
        active = self._cost_per_branch[:self.__index]
        if not active:
            return 0, 0
        return max(active), sum(active)
    
    # def solve(self, function):
    
    #     correct = (function(self) == self.__gold_answer)

        
    #     # sanity check
    #     if sum(self._cost_per_branch) != self.__cost:
    #         raise RuntimeError(
    #             f"Token accounting mismatch: total={self.__cost}, "
    #             f"per_branch_sum={sum(self._cost_per_branch)}"
    #         )
    #     return correct, self.__cost

    def solve(self,function):
        if not isinstance(function, BaseSolver):
            raise ValueError("The provided function is not callable.")
        return function.__call__(self)==self.__gold_answer, self.__cost


class ModelandTask:
    def __init__(self, model, dataset_name):
        self.model = model
        # self.data = data
        self.dataset_name = dataset_name    
        self.datas = json.load(open(f"{model}/{dataset_name}.json", 'r', encoding='utf-8'))
        self.data = [Question(info) for info in self.datas]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def evaluate(self, function, bootstrap_iter=1):
        accuracies = []
        costs = []
        sequential_costs = []
        
        def bootstrap(data):
            data = data.copy()
            data["each_branch"] = random.choices(data["each_branch"], k=len(data["each_branch"]))
            return data
        
        for i in range(bootstrap_iter):
            boot_strap_dat = [bootstrap(data) for data in self.datas]
            if bootstrap_iter == 1:
                boot_strap_dat = self.datas
            accs = []
            for _ in range(100):
                self.data = [Question(info,seed=_) for info in boot_strap_dat]
                total_cost = 0
                total_sequential_cost = 0
                correct_count = 0
                
                for question in self.data:
                    is_correct, cost = question.solve(function)
                    seq_tok, tot_tok = question.get_seq_and_total_tokens()
                    assert tot_tok == cost, f"Token cost mismatch: {tot_tok} vs {cost}"
                    if  type(function).__name__ == 'ESCSolver':
                        trunk = function.window_size
                        assert len(question.get_cost_per_branch_active()) % trunk == 0, f"Active branches {len(question.get_cost_per_branch_active())} not multiple of trunk {trunk}"
                        # calculate sequential cost for ESC
                        seq_tok = 0
                        active_costs = question.get_cost_per_branch_active()
                        # tokens within trunk are parallel but trunks are sequential
                        for i in range(0, len(active_costs), trunk):
                            seq_tok += max(active_costs[i:i+trunk])
                    total_cost += cost
                    total_sequential_cost += seq_tok
                    if is_correct:
                        correct_count += 1
                
                if len(self.data) > 0:
                    accs.append(correct_count / len(self.data))
                    costs.append(total_cost / len(self.data))
                    sequential_costs.append(total_sequential_cost / len(self.data))
                else:
                    accs.append(0)
                    costs.append(0)
                    sequential_costs.append(0)
            accuracies.append(np.mean(accs))
            # inner_std = np.std(accs)/np.sqrt(len(accs))
            # print(f"Inner std: {inner_std}")
        if bootstrap_iter > 1:
            std = np.std(accuracies)
        else:
            std = 0

        return {
            'method': function.description(),
            'accuracy': round(100 * sum(accuracies) / len(accuracies),2) if accuracies else 0,
            'avg_cost': sum(costs) / len(costs) if costs else 0,
            'avg_sequential_cost': sum(sequential_costs) / len(sequential_costs) if sequential_costs else 0,
            'std_accuracy': round(100 * std, 2)
        }
