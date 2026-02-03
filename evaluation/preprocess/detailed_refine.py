import json

def strip_latex_command(text, commands=['\\text', '\\box', '\\boxed', '\\textbf']):
    """
    去除指定的 latex 命令外壳，保留花括号内的内容。
    支持处理嵌套括号，例如 \text{A {B} C} -> A {B} C
    """
    if not isinstance(text, str):
        return text

    while True:
        found_something = False
        for cmd in commands:
            prefix = cmd + "{"
            start_idx = text.find(prefix)
            
            if start_idx != -1:
                found_something = True
                # 开始寻找匹配的右括号
                balance = 1
                content_start = start_idx + len(prefix)
                current_idx = content_start
                content_end = -1
                
                # 遍历字符串寻找闭合括号
                while current_idx < len(text):
                    char = text[current_idx]
                    if char == '{':
                        balance += 1
                    elif char == '}':
                        balance -= 1
                    
                    if balance == 0:
                        content_end = current_idx
                        break
                    current_idx += 1
                
                if content_end != -1:
                    # 提取内部内容
                    inner_content = text[content_start:content_end]
                    # 替换原字符串：头部 + 内部内容 + 尾部
                    text = text[:start_idx] + inner_content + text[content_end+1:]
                else:
                    # 如果没有找到匹配的闭合括号（格式错误的 LaTeX），
                    # 为防止死循环，我们跳过这个命令或者直接返回当前结果
                    # 这里选择跳过处理这个特定的起始点（实际业务中可能需要报错）
                    break 
        
        # 如果这一轮循环没有发现任何命令，说明处理完毕
        if not found_something:
            break
    if 'no' in text.lower():
        return "No"
    if "=" in text:
        return text.split('=')[-1].strip()
    if "is" in text:
        return text.split('is')[-1].strip()
    return text.replace('dfrac', 'frac')


def clean_data_list(input_list):
    # ---------------------------------------------------------
    # 第一步：去除尾部的 None
    # ---------------------------------------------------------
    # 我们创建一个副本以避免修改原列表，也可以选择原地修改
    # 从后往前找，找到第一个非 None 的值的索引
    last_valid_index = -1
    for i in range(len(input_list) - 1, -1, -1):
        if input_list[i] is not None:
            last_valid_index = i
            break
    
    # 切片截取有效部分（如果全是 None，last_valid_index 是 -1，切片 [:0] 为空列表，逻辑正确）
    cleaned_list = input_list[:last_valid_index + 1]

    # ---------------------------------------------------------
    # 第二步：处理 \text{} 和 \box{} (支持嵌套)
    # ---------------------------------------------------------

    # 对列表中的每一项应用清洗函数
    # 注意：列表中可能还有 None（原本在中间的），需要处理吗？
    # 根据你的描述只filter掉尾部的None，中间的None保持原样或转空字符串，这里假设保持原样
    result = []
    for item in cleaned_list:
        if item is None:
            result.append(None)
        else:
            result.append(strip_latex_command(item))
            
    return result
for model_name in ['Qwen3-0.6B-128', 'Qwen3-1.7B-128']:
    for dataset_name in ['aime24', 'aime25', 'hmmt25']:
        with open(f"{model_name}/{dataset_name}.json", 'r', encoding='utf-8') as f:
            datas=json.load(f)

        for data in datas:
            new_each_branch = []
            for branch in data['each_branch']:
                probe_matrix_mxn, branch_tokens, final_answer = branch

                new_each_branch.append( (clean_data_list(probe_matrix_mxn), branch_tokens, strip_latex_command(final_answer)) )
            data['each_branch'] = new_each_branch
            data['final_answers_trace'] = [strip_latex_command(ans) for ans in data['final_answers_trace']]
            data['gold_answer']= strip_latex_command(data['gold_answer'])
            # break

        json.dump(datas, open(f"{model_name}/{dataset_name}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)