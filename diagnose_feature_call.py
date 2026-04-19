# 特征提取调用追踪脚本
import ast
import os

def trace_feature_extraction():
    """追踪特征提取调用链"""
    print("=" * 60)
    print("🔍 特征提取调用链追踪")
    print("=" * 60)

    print("查找特征提取相关调用:")
    print("-" * 60)

    total_files = 0
    total_matches = 0

    # 遍历所有Python文件
    for root, dirs, files in os.walk('.'):
        # 跳过某些目录
        dirs[:] = [d for d in dirs if d not in ['__pycache__', '.git', 'cache', 'outputs']]

        for file in files:
            if file.endswith('.py'):
                total_files += 1
                filepath = os.path.join(root, file)

                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()

                        # 查找特征提取相关的代码
                        lines = content.split('\n')
                        matches = []

                        for i, line in enumerate(lines):
                            line_lower = line.lower()
                            # 查找相关关键词
                            if any(keyword in line_lower for keyword in ['extract_features', 'feature', 'snapshot', 'timeslice', 'rescale']):
                                # 排除注释和空行
                                if not line.strip().startswith('#') and line.strip():
                                    matches.append((i+1, line.strip()))

                        if matches:
                            total_matches += len(matches)
                            print(f"\n📄 文件: {filepath}")
                            for line_num, line_content in matches[:10]:  # 最多显示10个匹配
                                print(f"  行{line_num}: {line_content}")
                            if len(matches) > 10:
                                print(f"  ... 还有{len(matches)-10}个匹配")

                except Exception as e:
                    print(f"\n⚠️  读取文件失败: {filepath}")
                    print(f"  错误: {e}")

    print("\n" + "=" * 60)
    print("追踪完成")
    print("-" * 60)
    print(f"总文件数: {total_files}")
    print(f"总匹配数: {total_matches}")


def find_feature_functions():
    """查找特征提取相关函数"""
    print("\n" + "=" * 60)
    print("🔧 特征提取函数定位")
    print("=" * 60)

    for root, dirs, files in os.walk('.'):
        dirs[:] = [d for d in dirs if d not in ['__pycache__', '.git', 'cache', 'outputs']]

        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)

                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()

                        # 查找函数定义
                        lines = content.split('\n')
                        in_function = False
                        function_lines = []
                        function_name = ""

                        for i, line in enumerate(lines):
                            if line.strip().startswith('def'):
                                # 检查是否是特征相关函数
                                if 'feature' in line.lower() or 'extract' in line.lower():
                                    if in_function:
                                        # 打印之前的函数
                                        print(f"\n📄 文件: {filepath}")
                                        print(f"  函数: {function_name}")
                                        for func_line in function_lines[:10]:
                                            print(f"    {func_line}")
                                        if len(function_lines) > 10:
                                            print(f"    ... 还有{len(function_lines)-10}行")

                                    # 开始新函数
                                    in_function = True
                                    function_name = line.strip()
                                    function_lines = [line.strip()]
                                else:
                                    # 不是特征相关函数，重置
                                    in_function = False
                            elif in_function:
                                function_lines.append(line.rstrip())

                        # 打印最后一个函数
                        if in_function and ('feature' in function_name.lower() or 'extract' in function_name.lower()):
                            print(f"\n📄 文件: {filepath}")
                            print(f"  函数: {function_name}")
                            for func_line in function_lines[:10]:
                                print(f"    {func_line}")
                            if len(function_lines) > 10:
                                print(f"    ... 还有{len(function_lines)-10}行")

                except Exception as e:
                    pass


def analyze_snapshot_creation():
    """分析快照创建过程"""
    print("\n" + "=" * 60)
    print("📊 快照创建过程分析")
    print("=" * 60)

    for root, dirs, files in os.walk('.'):
        dirs[:] = [d for d in dirs if d not in ['__pycache__', '.git', 'cache', 'outputs']]

        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)

                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()

                        # 查找快照相关代码
                        if 'snapshot' in content.lower() and 'build' in content.lower():
                            print(f"\n📄 文件: {filepath}")
                            lines = content.split('\n')

                            for i, line in enumerate(lines):
                                if 'snapshot' in line.lower() and ('build' in line.lower() or 'create' in line.lower() or 'features' in line.lower()):
                                    print(f"  行{i+1}: {line.strip()}")

                except Exception as e:
                    pass


if __name__ == "__main__":
    trace_feature_extraction()
    find_feature_functions()
    analyze_snapshot_creation()

    print("\n" + "=" * 60)
    print("🎯 分析完成")
    print("=" * 60)
    print("接下来需要：")
    print("1. 检查特征提取函数是否接收时间片索引参数")
    print("2. 确认每个时间片是否独立计算特征")
    print("3. 验证特征随时间片变化")