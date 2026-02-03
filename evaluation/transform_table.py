
import pandas as pd
import os

def extract_n64_data(df):
    results = {}
    # 定义匹配配置: (Key, Solver, Strategy, 优先查找的列名)
    configs = [
        ('SC', 'MajorityVoteSolver', 'FullReadStrategy', 'solver_n'),
        ('ASC', 'ASCSolver', 'FullReadStrategy', 'solver_n'),
        ('ESC', 'ESCSolver', 'FullReadStrategy', 'solver_n'),
        ('SAC', 'MajorityVoteSolver', 'ConvergenceProbeStrategy', 'solver_n'),
        ('Probe', 'GreedySolver', 'ParallelESTPruningBurstMajorityOnActiveBranches', 'num_chains')
    ]
    
    for key, solv, strat, pref_n_col in configs:
        # 确定当前 CSV 中实际存在的列名
        actual_n_col = None
        for col in [pref_n_col, 'solver_n', 'n', 'num_chains']:
            if col in df.columns:
                actual_n_col = col
                break
        
        if not actual_n_col:
            results[key] = (None, None, None)
            continue

        # 核心匹配逻辑：Solver + Strategy + n=64
        mask = (df['Solver'] == solv) & (df['Strategy'] == strat)
        # 强制转换为 float 比较，确保 64 和 64.0 都能匹配
        row = df[mask & (df[actual_n_col].astype(float) == 64.0)]
        
        if not row.empty:
            # 针对 ParallelESTPruning 数据集，如果有多个 64 (比如不同的 prune_patience)，取第一行
            target_row = row.iloc[0] 
            
            acc = float(target_row['Acc'])
            tok_val = float(target_row['Cost'])
            
            # 逻辑修正：ASC 的 Seq 等于 Total
            if key == 'ASC':
                seq_val = tok_val
            else:
                seq_val = float(target_row['Seq_Cost']) if 'Seq_Cost' in target_row.index else tok_val
            
            results[key] = (acc, seq_val / 1000.0, tok_val / 1000.0)
        else:
            results[key] = (None, None, None)
    return results

def format_change(current, baseline):
    if current is None or baseline is None or baseline == 0:
        return ""
    diff = (current - baseline) / baseline * 100
    if abs(diff) < 0.01: return ""
    # Token 减少是好事(红色)，增加是坏事(绿色)
    color = "red" if diff < 0 else "green!80!black"
    sign = "+" if diff > 0 else ""
    return f"\\textsuperscript{{\\scriptsize(\\textcolor{{{color}}}{{{sign}{diff:.1f}\\%}})}}"

def main():
    models = ["Qwen3-0.6B-128", "Qwen3-1.7B-128", "Qwen3-4B-128", "Qwen3-8B-128"]
    datasets = ["aime24", "aime25", "hmmt25"]
    root_dir = "./"
    output_filename = "table.tex"

    latex_lines = []
    latex_lines.append(r"\begin{table*}[t]")
    latex_lines.append(r"\centering\small")
    latex_lines.append(r"\begin{adjustbox}{width=\textwidth}")
    latex_lines.append(r"\begin{tabular}{l l ccc ccc ccc ccc}")
    latex_lines.append(r"\toprule")
    latex_lines.append(r"Method & Type & \multicolumn{3}{c}{AIME24} & \multicolumn{3}{c}{AIME25} & \multicolumn{3}{c}{HMMT25} & \multicolumn{3}{c}{Avg.} \\")
    latex_lines.append(r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}\cmidrule(lr){12-14}")
    latex_lines.append(r"& & Acc. $\uparrow$ & SeqTokens $\downarrow$ & Tokens $\downarrow$ & Acc. $\uparrow$ & SeqTokens $\downarrow$ & Tokens $\downarrow$ & Acc. $\uparrow$ & SeqTokens $\downarrow$ & Tokens $\downarrow$ & Acc. $\uparrow$ & SeqTokens $\downarrow$ & Tokens $\downarrow$ \\")

    for model in models:
        latex_lines.append(r"\midrule")
        # 移除加粗
        latex_lines.append(f"\\multicolumn{{14}}{{l}}{{\\textit{{Base Model: {model}}}}} \\\\")
        latex_lines.append(r"\midrule")
        
        methods = [
            ("SC @ 64", "SC", "Parallel"),
            ("ASC", "ASC", "Seq."),
            ("ESC", "ESC", "Hybrid"),
            ("SC @ 64 + SAC", "SAC", "Parallel"),
            ("Parallel-Probe", "Probe", "Parallel")
        ]
        
        model_data = {}
        # 兼容两种文件夹命名
        path_suffix = "-128" if os.path.exists(os.path.join(root_dir, f"matrix_results_{model}-128")) else ""

        for label, m_key, m_type in methods:
            acc_list, seq_list, tok_list, raw_cells = [], [], [], []
            for ds in datasets:
                file_path = os.path.join(root_dir, f"matrix_results_{model}{path_suffix}", f"{ds}_raw.csv")
                if os.path.exists(file_path):
                    res = extract_n64_data(pd.read_csv(file_path))
                    acc, seq, tok = res.get(m_key, (None, None, None))
                    if acc is not None:
                        raw_cells.extend([f"{acc:.1f}", f"{seq:.1f}k", f"{tok:.1f}k"])
                        acc_list.append(acc); seq_list.append(seq); tok_list.append(tok)
                    else: raw_cells.extend(["--", "--", "--"])
                else: raw_cells.extend(["--", "--", "--"])
            
            if acc_list:
                model_data[m_key] = {
                    'acc': sum(acc_list)/len(acc_list),
                    'seq': sum(seq_list)/len(seq_list),
                    'tok': sum(tok_list)/len(tok_list),
                    'raw_cells': raw_cells
                }

        # 基准 (SC @ 64)
        base_seq = model_data['SC']['seq'] if 'SC' in model_data else None
        base_tok = model_data['SC']['tok'] if 'SC' in model_data else None

        for label, m_key, m_type in methods:
            if m_key not in model_data: continue
            d = model_data[m_key]
            
            # Avg 列逻辑
            seq_chg = format_change(d['seq'], base_seq) if m_key != 'SC' else ""
            tok_chg = format_change(d['tok'], base_tok) if m_key != 'SC' else ""
            
            row = [label, m_type] + d['raw_cells'] + [
                f"{d['acc']:.1f}", 
                f"{d['seq']:.1f}k {seq_chg}", 
                f"{d['tok']:.1f}k {tok_chg}"
            ]
            latex_lines.append(" & ".join(row) + r" \\")

    latex_lines.append(r"\bottomrule \end{tabular} \end{adjustbox} \end{table*}")

    with open(output_filename, "w", encoding="utf-8") as f:
        f.write("\n".join(latex_lines))
    print(f"完成！数据已更新至 {output_filename}")

if __name__ == "__main__":
    main()