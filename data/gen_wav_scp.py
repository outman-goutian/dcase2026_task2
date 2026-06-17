#!/usr/bin/env python3
"""生成wav文件的scp文件
用法: python gen_wav_scp.py <文件夹路径> <输出scp文件名> <train|test>
"""

import os
import sys
import argparse


def extract_label(basename, split='train'):
    """从文件名提取属性标签，参考gene_scp.py的逻辑"""
    # 去除 ".wav" 后缀，并按下划线 "_" 分割字符串
    parts = basename.split('.wav')[0].split('_')

    # final test 文件名通常是 section_00_0000.wav，没有 domain/status 字段。
    # 补成当前 validation loader 可解析的格式。
    if split == 'test' and len(parts) == 3 and parts[0] == 'section':
        return f"{parts[0]}_{parts[1]}_source_test_normal_{parts[2]}"

    # 取前5段 + 从第7段开始的内容（跳过第6段）
    labels = parts[:5] + parts[6:]

    # 将分割后的列表用下划线重新拼接成标签字符串
    label = '_'.join(labels)

    return label


def find_wavs(root_dir, split='train', machines=None):
    """递归查找root_dir下所有指定目录(split)中的wav文件，返回 (子文件夹名, wav路径列表) 的字典"""
    result = {}
    machine_set = set(machines) if machines else None

    for dirpath, dirnames, filenames in os.walk(root_dir):
        if os.path.basename(dirpath) == split:
            # 获取当前目录对应的子文件夹名
            # 从root_dir到当前目录的相对路径的第一个层级
            rel_path = os.path.relpath(dirpath, root_dir)
            sub_folder = rel_path.split(os.sep)[0]
            if machine_set is not None and sub_folder not in machine_set:
                continue

            for filename in filenames:
                if filename.endswith('.wav'):
                    if sub_folder not in result:
                        result[sub_folder] = []
                    result[sub_folder].append(os.path.join(dirpath, filename))

    # 对每个子文件夹的wav文件排序
    for sub_folder in result:
        result[sub_folder] = sorted(result[sub_folder])

    return result


def main():
    parser = argparse.ArgumentParser(description='生成wav文件的scp文件')
    parser.add_argument('root_dir', help='数据根目录路径')
    parser.add_argument('output_file', help='输出scp文件名')
    parser.add_argument('split', nargs='?', default='train', choices=['train', 'test'],
                        help='指定train或test (默认: train)')
    parser.add_argument(
        '--path-prefix',
        default=None,
        help='输出路径前缀；例如 /workspace/data。默认写入本机绝对路径'
    )
    parser.add_argument(
        '--machines',
        nargs='+',
        default=None,
        help='只生成指定机器类型，例如 --machines ToyCar fan'
    )

    args = parser.parse_args()

    root_dir = args.root_dir
    output_file = args.output_file
    split = args.split

    # 处理相对路径
    root_dir = os.path.abspath(root_dir)

    if not os.path.isdir(root_dir):
        print(f"错误: 目录不存在: {root_dir}")
        sys.exit(1)

    # 查找所有wav文件，按子文件夹分组
    wav_dict = find_wavs(root_dir, split, args.machines)

    if not wav_dict:
        print(f"警告: 在 {root_dir} 下没有找到任何{split}目录中的wav文件")
        with open(output_file, 'w') as f:
            pass
        print(f"已创建空的scp文件: {output_file}")
        return

    # 写入scp文件
    total_count = 0
    with open(output_file, 'w') as f:
        for sub_folder, wav_files in sorted(wav_dict.items()):
            for wav_path in wav_files:
                basename = os.path.basename(wav_path)
                label = extract_label(basename, split)
                full_label = f"{sub_folder}_{label}"
                if args.path_prefix:
                    rel_path = os.path.relpath(wav_path, root_dir)
                    output_path = os.path.join(args.path_prefix, rel_path)
                else:
                    output_path = os.path.abspath(wav_path)
                f.write(f"{full_label}\t{output_path}\n")
                total_count += 1

    print(f"已生成scp文件: {output_file}")
    print(f"共处理 {total_count} 个wav文件，分布在 {len(wav_dict)} 个子文件夹中")


if __name__ == "__main__":
    main()
