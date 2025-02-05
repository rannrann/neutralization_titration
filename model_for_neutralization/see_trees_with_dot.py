import os
from graphviz import Source

# 确保路径存在
output_dir = "graph/trees/correct_trees_with_7_parameters"
os.makedirs(output_dir, exist_ok=True)  # 如果路径不存在，则创建路径

# 读取 .dot 文件并渲染
for i in [14, 22, 30, 89]:
    dot_file = f"model_for_neutralization/correct_trees_with_7_parameters/correct_tree_{i}.dot"
    graph = Source.from_file(dot_file)

    # 渲染并保存为 PNG 图像
    output_file = os.path.join(output_dir, f"correct_tree_{i}")
    graph.render(output_file, format="png", cleanup=True)
    print(f"Visualization saved as {output_file}.png")


