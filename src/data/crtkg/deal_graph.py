"""
GraphRAG 三元组提取工具
从 GraphRAG 输出文件中提取实体和关系，保存为 JSON 格式
"""

import pandas as pd
import json
import os
from typing import List, Dict, Any
from datetime import datetime


class GraphRAGExtractor:
    """从 GraphRAG 输出中提取三元组"""

    def __init__(self, input_dir: str, output_dir: str = "./extracted_data"):
        """
        初始化提取器

        Args:
            input_dir: GraphRAG 输出目录路径
            output_dir: 提取结果保存目录
        """
        self.input_dir = input_dir
        self.output_dir = output_dir

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        print(f"输入目录: {input_dir}")
        print(f"输出目录: {output_dir}")

    def load_entities(self) -> pd.DataFrame:
        """加载实体数据"""
        file_path = os.path.join(self.input_dir, "entities.parquet")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"找不到文件: {file_path}")

        df = pd.read_parquet(file_path)
        print(f"✓ 加载 {len(df)} 个实体")
        return df

    def load_relationships(self) -> pd.DataFrame:
        """加载关系数据"""
        file_path = os.path.join(self.input_dir, "relationships.parquet")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"找不到文件: {file_path}")

        df = pd.read_parquet(file_path)
        print(f"✓ 加载 {len(df)} 个关系")
        return df

    def load_communities(self) -> pd.DataFrame:
        """加载社区数据（可选）"""
        file_path = os.path.join(self.input_dir, "communities.parquet")
        if os.path.exists(file_path):
            df = pd.read_parquet(file_path)
            print(f"✓ 加载 {len(df)} 个社区")
            return df
        else:
            print("⚠ 未找到社区数据文件")
            return pd.DataFrame()

    def extract_entities(self, entities_df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """
        提取实体详细信息

        Returns:
            字典格式: {entity_id: {name, type, description, ...}}
        """
        entity_dict = {}

        for _, row in entities_df.iterrows():
            entity_id = str(row.get('id', row.get('human_readable_id', '')))

            entity_dict[entity_id] = {
                'id': entity_id,
                'name': str(row.get('name', row.get('title', ''))),
                'type': str(row.get('type', 'ENTITY')),
                'description': str(row.get('description', '')),
                'degree': int(row.get('degree', 0)) if pd.notna(row.get('degree')) else 0,
                'community_ids': self._safe_list(row.get('community_ids', [])),
                'text_unit_ids': self._safe_list(row.get('text_unit_ids', [])),
            }

        print(f"✓ 提取 {len(entity_dict)} 个实体详情")
        return entity_dict

    def extract_triples(self,
                        relationships_df: pd.DataFrame,
                        entity_dict: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        提取三元组

        Returns:
            列表格式: [{subject, predicate, object, ...}]
        """
        triples = []

        for idx, row in relationships_df.iterrows():
            source_id = str(row.get('source', ''))
            target_id = str(row.get('target', ''))

            # 获取实体名称
            source_name = entity_dict.get(source_id, {}).get('name', source_id)
            target_name = entity_dict.get(target_id, {}).get('name', target_id)

            triple = {
                'id': f"rel_{idx}",
                'subject': source_name,
                'subject_id': source_id,
                'predicate': str(row.get('type', row.get('description', 'RELATED_TO'))),
                'object': target_name,
                'object_id': target_id,
                'weight': float(row.get('weight', 1.0)) if pd.notna(row.get('weight')) else 1.0,
                'description': str(row.get('description', '')),
                'source_degree': int(row.get('source_degree', 0)) if pd.notna(row.get('source_degree')) else 0,
                'target_degree': int(row.get('target_degree', 0)) if pd.notna(row.get('target_degree')) else 0,
                'rank': int(row.get('rank', 0)) if pd.notna(row.get('rank')) else 0,
            }

            triples.append(triple)

        print(f"✓ 提取 {len(triples)} 个三元组")
        return triples

    def _safe_list(self, value):
        """安全转换为列表"""
        import numpy as np

        # 检查是否为 None 或 pandas NA
        if value is None:
            return []

        # 处理 numpy 数组的情况
        if isinstance(value, np.ndarray):
            if value.size == 0:  # 空数组
                return []
            return value.tolist()  # 转换为 Python 列表

        # 检查是否为 pandas 标量 NA (NaN, NaT, None)
        if pd.isna(value) and not hasattr(value, '__len__'):
            return []

        if isinstance(value, list):
            return value
        if isinstance(value, str):
            return [value]
        if isinstance(value, (tuple, set)):
            return list(value)

        return []

    def analyze_statistics(self, entities: Dict, triples: List) -> Dict[str, Any]:
        """分析图谱统计信息"""
        # 实体类型统计
        entity_types = {}
        for entity in entities.values():
            etype = entity['type']
            entity_types[etype] = entity_types.get(etype, 0) + 1

        # 关系类型统计
        relation_types = {}
        for triple in triples:
            rtype = triple['predicate']
            relation_types[rtype] = relation_types.get(rtype, 0) + 1

        # 度数统计
        degrees = [e['degree'] for e in entities.values() if e['degree'] > 0]
        avg_degree = sum(degrees) / len(degrees) if degrees else 0
        max_degree = max(degrees) if degrees else 0

        # 权重统计
        weights = [t['weight'] for t in triples]
        avg_weight = sum(weights) / len(weights) if weights else 0

        stats = {
            'total_entities': len(entities),
            'total_triples': len(triples),
            'entity_types': entity_types,
            'relation_types': relation_types,
            'degree_stats': {
                'average': round(avg_degree, 2),
                'max': max_degree,
                'min': min(degrees) if degrees else 0
            },
            'weight_stats': {
                'average': round(avg_weight, 2),
                'max': max(weights) if weights else 0,
                'min': min(weights) if weights else 0
            }
        }

        return stats

    def print_statistics(self, stats: Dict[str, Any]):
        """打印统计信息"""
        print("\n" + "=" * 60)
        print("图谱统计信息")
        print("=" * 60)
        print(f"实体总数: {stats['total_entities']}")
        print(f"三元组总数: {stats['total_triples']}")

        print(f"\n实体类型分布 (Top 10):")
        for etype, count in sorted(stats['entity_types'].items(),
                                   key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {etype}: {count}")

        print(f"\n关系类型分布 (Top 10):")
        for rtype, count in sorted(stats['relation_types'].items(),
                                   key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {rtype}: {count}")

        print(f"\n度数统计:")
        print(f"  平均度数: {stats['degree_stats']['average']}")
        print(f"  最大度数: {stats['degree_stats']['max']}")
        print(f"  最小度数: {stats['degree_stats']['min']}")

        print(f"\n权重统计:")
        print(f"  平均权重: {stats['weight_stats']['average']}")
        print(f"  最大权重: {stats['weight_stats']['max']}")
        print(f"  最小权重: {stats['weight_stats']['min']}")
        print("=" * 60)

    def print_sample_data(self, entities: Dict, triples: List, n: int = 5):
        """打印示例数据"""
        print("\n" + "=" * 60)
        print(f"示例实体 (前 {n} 个)")
        print("=" * 60)
        for i, (eid, entity) in enumerate(list(entities.items())[:n], 1):
            print(f"{i}. {entity['name']} ({entity['type']})")
            print(f"   ID: {entity['id']}")
            print(f"   度数: {entity['degree']}")
            if entity['description']:
                desc = entity['description'][:100] + "..." if len(entity['description']) > 100 else entity[
                    'description']
                print(f"   描述: {desc}")
            print()

        print("=" * 60)
        print(f"示例三元组 (前 {n} 个)")
        print("=" * 60)
        for i, triple in enumerate(triples[:n], 1):
            print(f"{i}. ({triple['subject']}) --[{triple['predicate']}]--> ({triple['object']})")
            print(f"   权重: {triple['weight']}")
            if triple['description']:
                desc = triple['description'][:100] + "..." if len(triple['description']) > 100 else triple[
                    'description']
                print(f"   描述: {desc}")
            print()
        print("=" * 60)

    def save_to_json(self,
                     entities: Dict,
                     triples: List,
                     stats: Dict,
                     filename: str = "graph_data.json"):
        """保存为单个 JSON 文件"""
        output_path = os.path.join(self.output_dir, filename)

        data = {
            'metadata': {
                'extraction_time': datetime.now().isoformat(),
                'source_directory': self.input_dir,
                'statistics': stats
            },
            'entities': entities,
            'triples': triples
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"\n✓ 数据已保存到: {output_path}")
        print(f"  文件大小: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")

        return output_path

    def save_separate_files(self,
                            entities: Dict,
                            triples: List,
                            stats: Dict):
        """保存为多个独立的 JSON 文件"""
        # 保存实体
        entities_path = os.path.join(self.output_dir, "entities.json")
        with open(entities_path, 'w', encoding='utf-8') as f:
            json.dump(entities, f, ensure_ascii=False, indent=2)
        print(f"✓ 实体已保存到: {entities_path}")

        # 保存三元组
        triples_path = os.path.join(self.output_dir, "triples.json")
        with open(triples_path, 'w', encoding='utf-8') as f:
            json.dump(triples, f, ensure_ascii=False, indent=2)
        print(f"✓ 三元组已保存到: {triples_path}")

        # 保存统计信息
        stats_path = os.path.join(self.output_dir, "statistics.json")
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"✓ 统计信息已保存到: {stats_path}")

        return entities_path, triples_path, stats_path

    def run(self,
            save_mode: str = "single",  # "single" 或 "separate"
            show_samples: bool = True,
            sample_size: int = 5):
        """
        执行完整的提取流程

        Args:
            save_mode: "single" 保存为单个文件, "separate" 保存为多个文件
            show_samples: 是否显示示例数据
            sample_size: 示例数据数量
        """
        print("\n" + "=" * 60)
        print("开始提取 GraphRAG 数据")
        print("=" * 60 + "\n")

        # 1. 加载数据
        print("步骤 1: 加载数据文件")
        entities_df = self.load_entities()
        relationships_df = self.load_relationships()

        # 可选：加载社区数据
        try:
            communities_df = self.load_communities()
        except Exception as e:
            print(f"⚠ 加载社区数据失败: {e}")
            communities_df = pd.DataFrame()

        # 2. 提取结构化数据
        print("\n步骤 2: 提取结构化数据")
        entities = self.extract_entities(entities_df)
        triples = self.extract_triples(relationships_df, entities)

        # 3. 统计分析
        print("\n步骤 3: 分析统计信息")
        stats = self.analyze_statistics(entities, triples)
        self.print_statistics(stats)

        # 4. 显示示例
        if show_samples:
            self.print_sample_data(entities, triples, n=sample_size)

        # 5. 保存数据
        print("\n步骤 4: 保存数据")
        if save_mode == "single":
            output_file = self.save_to_json(entities, triples, stats)
            return output_file
        else:
            output_files = self.save_separate_files(entities, triples, stats)
            return output_files


def main():
    """主函数"""
    # 配置路径
    INPUT_DIR = "./output"  # GraphRAG 输出目录
    OUTPUT_DIR = "./extracted_data"  # 提取结果保存目录

    # 创建提取器
    extractor = GraphRAGExtractor(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR
    )

    # 执行提取
    # 模式1: 保存为单个 JSON 文件 (推荐用于后续导入 Neo4j)
    output_file = extractor.run(
        save_mode="single",  # 或 "separate" 保存为多个文件
        show_samples=True,  # 显示示例数据
        sample_size=5  # 示例数量
    )

    print("\n" + "=" * 60)
    print("✅ 提取完成！")
    print("=" * 60)
    print(f"\n生成的文件可用于下一步导入 Neo4j 数据库")
    print(f"请运行: python import_to_neo4j.py")


if __name__ == "__main__":
    main()