"""
Neo4j 导入工具 (最终修复版)
修复了 ID 类型不匹配的问题：从匹配 UUID 改为匹配 Name
"""

import json
import os
from neo4j import GraphDatabase
from typing import List, Dict, Any
import time


class Neo4jImporter:
    """Neo4j 数据导入器"""

    def __init__(self, uri: str, user: str, password: str):
        """
        初始化 Neo4j 连接

        Args:
            uri: Neo4j 连接地址 (例如: bolt://localhost:7687)
            user: 用户名
            password: 密码
        """
        self.uri = uri
        self.user = user
        self.driver = None

        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            # 测试连接
            with self.driver.session() as session:
                result = session.run("RETURN 1 as test")
                result.single()
            print(f"✓ 成功连接到 Neo4j: {uri}")
        except Exception as e:
            print(f"✗ 连接 Neo4j 失败: {e}")
            raise

    def close(self):
        """关闭数据库连接"""
        if self.driver:
            self.driver.close()
            print("✓ 数据库连接已关闭")

    def load_json_data(self, json_file: str) -> Dict[str, Any]:
        """
        从 JSON 文件加载数据

        Args:
            json_file: JSON 文件路径

        Returns:
            包含 entities 和 triples 的字典
        """
        if not os.path.exists(json_file):
            raise FileNotFoundError(f"找不到文件: {json_file}")

        print(f"\n加载数据文件: {json_file}")

        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        entities = data.get('entities', {})
        triples = data.get('triples', [])
        metadata = data.get('metadata', {})

        print(f"✓ 加载 {len(entities)} 个实体")
        print(f"✓ 加载 {len(triples)} 个三元组")

        if metadata:
            stats = metadata.get('statistics', {})
            print(f"\n数据统计:")
            print(f"  实体总数: {stats.get('total_entities', 0)}")
            print(f"  三元组总数: {stats.get('total_triples', 0)}")

        return {
            'entities': entities,
            'triples': triples,
            'metadata': metadata
        }

    def clear_database(self, confirm: bool = True):
        """
        清空数据库（危险操作！）

        Args:
            confirm: 必须设置为 True 才能执行
        """
        if not confirm:
            print("⚠ 清空数据库需要确认，请设置 confirm=True")
            return

        print("\n⚠ 警告: 即将清空数据库中的所有数据！")
        user_input = input("确认要继续吗？ (输入 'YES' 确认): ")

        if user_input != "YES":
            print("✓ 操作已取消")
            return

        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            print("✓ 数据库已清空")

    def create_constraints_and_indexes(self):
        """创建约束和索引以提高性能"""
        print("\n创建约束和索引...")

        with self.driver.session() as session:
            # 为实体ID创建唯一约束
            try:
                session.run("""
                    CREATE CONSTRAINT entity_id IF NOT EXISTS
                    FOR (e:Entity) REQUIRE e.id IS UNIQUE
                """)
                print("✓ 创建实体ID唯一约束")
            except Exception as e:
                print(f"⚠ 创建约束失败 (可能已存在): {e}")

            # 为实体名称创建索引 (关键：提高按名称查找的速度)
            try:
                session.run("""
                    CREATE INDEX entity_name IF NOT EXISTS
                    FOR (e:Entity) ON (e.name)
                """)
                print("✓ 创建实体名称索引")
            except Exception as e:
                print(f"⚠ 创建索引失败 (可能已存在): {e}")

            # 为实体类型创建索引
            try:
                session.run("""
                    CREATE INDEX entity_type IF NOT EXISTS
                    FOR (e:Entity) ON (e.type)
                """)
                print("✓ 创建实体类型索引")
            except Exception as e:
                print(f"⚠ 创建索引失败 (可能已存在): {e}")

    def import_entities(self, entities: Dict[str, Dict[str, Any]], batch_size: int = 1000):
        """
        批量导入实体
        """
        print(f"\n导入实体 (批大小: {batch_size})...")

        # 转换字典为列表
        entities_list = []
        for entity_id, entity_data in entities.items():
            # 确保 id 字段存在
            entity_data['id'] = entity_id
            entities_list.append(entity_data)

        total_batches = (len(entities_list) + batch_size - 1) // batch_size

        start_time = time.time()

        with self.driver.session() as session:
            for i in range(0, len(entities_list), batch_size):
                batch = entities_list[i:i+batch_size]
                batch_num = i // batch_size + 1

                session.run("""
                    UNWIND $entities AS entity
                    MERGE (e:Entity {id: entity.id})
                    SET e.name = entity.name,
                        e.type = entity.type,
                        e.description = entity.description,
                        e.degree = entity.degree,
                        e.community_ids = entity.community_ids,
                        e.text_unit_ids = entity.text_unit_ids
                """, entities=batch)

                print(f"  批次 {batch_num}/{total_batches} 完成 ({len(batch)} 个实体)")

        elapsed_time = time.time() - start_time
        print(f"✓ 导入 {len(entities_list)} 个实体完成 (耗时: {elapsed_time:.2f}秒)")

    def import_relationships_without_apoc(self, triples: List[Dict[str, Any]], batch_size: int = 1000):
        """
        不使用 APOC 插件导入关系
        【核心修复】：使用 name 字段匹配节点，而不是 id
        """
        print(f"\n导入关系 (批大小: {batch_size})...")

        if not triples:
            print("⚠ 警告: 三元组列表为空！")
            return

        # ========== 1. 智能检测字段名 ==========
        sample = triples[0]
        print(f"\n📋 检测三元组数据结构...")
        print(f"   原始数据键: {list(sample.keys())}")

        subject_key = None
        object_key = None

        possible_subject_keys = ['subject_id', 'subject', 'source', 'head']
        possible_object_keys = ['object_id', 'object', 'target', 'tail']

        for key in possible_subject_keys:
            if key in sample:
                subject_key = key
                break

        for key in possible_object_keys:
            if key in sample:
                object_key = key
                break

        if not subject_key or not object_key:
            print(f"✗ 错误: 无法识别三元组中的ID字段名！")
            return

        print(f"✓ 自动匹配字段 -> Subject: '{subject_key}', Object: '{object_key}'")
        print(f"   样本数据: {sample.get(subject_key)} --[{sample.get('predicate')}]--> {sample.get(object_key)}")

        # ========== 2. 关键验证：检查是否应该按名称匹配 ==========
        # 如果 subject_id 看起来像 UUID，就用 id 匹配；否则用 name 匹配
        sample_id_val = str(sample[subject_key])
        # 简单的 UUID 启发式检查：长度大于20且包含连字符，或者看起来像哈希
        is_uuid_like = ('-' in sample_id_val and len(sample_id_val) > 20) or len(sample_id_val) == 32

        if is_uuid_like:
            print(f"   🔍 检测到 ID 格式类似 UUID，将使用 ID 匹配节点")
            match_field = "id"
        else:
            print(f"   🔍 检测到 ID 为文本名称，将使用 NAME 匹配节点 (修复方案)")
            match_field = "name"

        # ========== 3. 数据标准化与分组 ==========
        relations_by_type = {}

        for triple in triples:
            original_predicate = triple.get('predicate', 'RELATED_TO')
            rel_type = self._normalize_relationship_type(original_predicate)

            if rel_type not in relations_by_type:
                relations_by_type[rel_type] = []

            std_triple = {
                'subject_val': triple[subject_key], # 使用通用键名
                'object_val': triple[object_key],
                'weight': triple.get('weight', 0.0),
                'description': triple.get('description', ''),
                'source_degree': triple.get('source_degree', 0),
                'target_degree': triple.get('target_degree', 0),
                'rank': triple.get('rank', 0),
                'original_predicate': original_predicate
            }
            relations_by_type[rel_type].append(std_triple)

        print(f"  发现 {len(relations_by_type)} 种关系类型")

        # ========== 4. 验证节点是否存在 (使用确定的字段) ==========
        sample_rel_type = list(relations_by_type.keys())[0]
        sample_rel = relations_by_type[sample_rel_type][0]

        sample_sub_val = sample_rel['subject_val']
        sample_obj_val = sample_rel['object_val']

        print(f"\n🔍 验证样本节点 (使用 {match_field})...")
        with self.driver.session() as session:
            # 构建查询
            query = f"MATCH (n:Entity) WHERE n.{match_field} = $val RETURN count(n) as c"

            res = session.run(query, val=sample_sub_val)
            sub_count = res.single()['c']

            res = session.run(query, val=sample_obj_val)
            obj_count = res.single()['c']

            if sub_count == 0:
                print(f"   ⚠ 警告: 源节点 '{sample_sub_val}' 在数据库中不存在！")
            if obj_count == 0:
                print(f"   ⚠ 警告: 目标节点 '{sample_obj_val}' 在数据库中不存在！")

            if sub_count > 0 and obj_count > 0:
                print(f"   ✓ 样本节点验证通过，可以开始导入")

        # ========== 5. 批量导入 ==========
        start_time = time.time()
        total_imported = 0

        for rel_type, rel_triples in relations_by_type.items():
            print(f"\n  正在处理关系类型: {rel_type} ({len(rel_triples)} 条)")

            total_batches = (len(rel_triples) + batch_size - 1) // batch_size

            with self.driver.session() as session:
                for i in range(0, len(rel_triples), batch_size):
                    batch = rel_triples[i:i+batch_size]
                    batch_num = i // batch_size + 1

                    try:
                        # 使用动态构建的 Cypher，注意使用 {match_field}
                        # 关键点：n.{match_field} = triple.subject_val
                        query = f"""
                            UNWIND $triples AS triple
                            MATCH (source:Entity)
                            WHERE source.{match_field} = triple.subject_val
                            MATCH (target:Entity)
                            WHERE target.{match_field} = triple.object_val
                            MERGE (source)-[r:{rel_type}]->(target)
                            SET r.weight = triple.weight,
                                r.description = triple.description,
                                r.original_predicate = triple.original_predicate,
                                r.source_degree = triple.source_degree,
                                r.target_degree = triple.target_degree,
                                r.rank = triple.rank
                        """

                        result = session.run(query, triples=batch)
                        summary = result.consume()

                        count = summary.counters.relationships_created
                        total_imported += count

                        if count > 0:
                            print(f"    批次 {batch_num}/{total_batches}: 创建了 {count} 条关系")
                        else:
                            print(f"    批次 {batch_num}/{total_batches}: 跳过 (未找到节点)")

                    except Exception as e:
                        print(f"    ✗ 批次 {batch_num}/{total_batches} 失败: {e}")

        elapsed_time = time.time() - start_time
        print(f"\n✓ 关系导入完成。总计创建: {total_imported} 条 (耗时: {elapsed_time:.2f}秒)")

    def _normalize_relationship_type(self, predicate: str) -> str:
        """标准化关系类型名称"""
        if not predicate or not predicate.strip():
            return 'RELATED_TO'

        normalized = predicate.strip().replace(' ', '_')
        normalized = ''.join(c if c.isalnum() or c == '_' else '_' for c in normalized)
        normalized = normalized.upper()
        while '__' in normalized:
            normalized = normalized.replace('__', '_')
        normalized = normalized.strip('_')

        if not normalized or normalized.replace('_', '') == '':
            return 'RELATED_TO'

        if normalized[0].isdigit():
            normalized = 'REL_' + normalized

        return normalized

    def verify_import(self):
        """验证导入结果"""
        print("\n验证导入结果...")

        with self.driver.session() as session:
            # 统计节点数
            result = session.run("MATCH (n:Entity) RETURN count(n) as count")
            node_count = result.single()['count']

            # 统计关系数
            result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
            rel_count = result.single()['count']

            # 统计关系类型
            result = session.run("""
                MATCH ()-[r]->()
                RETURN type(r) as rel_type, count(r) as count
                ORDER BY count DESC
                LIMIT 10
            """)
            rel_types = list(result)

            print("\n" + "="*60)
            print("导入验证结果")
            print("="*60)
            print(f"节点总数: {node_count}")
            print(f"关系总数: {rel_count}")

            print(f"\n关系类型分布 (Top 10):")
            for record in rel_types:
                print(f"  {record['rel_type']}: {record['count']}")

            # 显示示例数据
            print(f"\n示例数据 (随机5条):")
            result = session.run("""
                MATCH (n:Entity)-[r]->(m:Entity)
                RETURN n.name as source, type(r) as relationship, 
                       m.name as target, r.weight as weight
                ORDER BY rand()
                LIMIT 5
            """)

            for i, record in enumerate(result, 1):
                print(f"{i}. ({record['source']}) -[{record['relationship']} (权重:{record['weight']:.2f})]-> ({record['target']})")

            print("="*60)


def main():
    """主函数"""

    # ========== 配置区域 ==========
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "jbh966225"

    JSON_FILE = "./extracted_data/graph_data.json"

    BATCH_SIZE = 1000
    CLEAR_DATABASE = True # 如果你之前已经导入了实体，这里设为 False 以免重新导实体
    USE_APOC = False
    # ==============================

    print("\n" + "="*60)
    print("开始导入数据到 Neo4j (修复版)")
    print("="*60)

    try:
        importer = Neo4jImporter(
            uri=NEO4J_URI,
            user=NEO4J_USER,
            password=NEO4J_PASSWORD
        )
    except Exception as e:
        print(f"\n✗ 初始化失败: {e}")
        return

    try:
        # 1. 加载数据
        print("\n步骤 1: 加载数据")
        data = importer.load_json_data(JSON_FILE)
        entities = data['entities']
        triples = data['triples']

        # 2. 可选：清空数据库
        if CLEAR_DATABASE:
            print("\n步骤 2: 清空数据库")
            importer.clear_database(confirm=True)

            # 3. 创建约束和索引
            print("\n步骤 3: 创建约束和索引")
            importer.create_constraints_and_indexes()

            # 4. 导入实体
            print("\n步骤 4: 导入实体")
            importer.import_entities(entities, batch_size=BATCH_SIZE)
        else:
            # 如果不清空数据库，实体应该已经存在，只需确保索引存在
            print("\n步骤 2: 检查/创建索引")
            importer.create_constraints_and_indexes()

        # 5. 导入关系
        print("\n步骤 5: 导入关系")
        importer.import_relationships_without_apoc(triples, batch_size=BATCH_SIZE)

        # 6. 验证导入
        print("\n步骤 6: 验证导入")
        importer.verify_import()

        print("\n" + "="*60)
        print("✅ 导入完成！")
        print("="*60)

    except Exception as e:
        print(f"\n✗ 导入失败: {e}")
        import traceback
        traceback.print_exc()
    finally:
        importer.close()


if __name__ == "__main__":
    main()
