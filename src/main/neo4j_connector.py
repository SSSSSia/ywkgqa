"""
Neo4j 数据库连接模块 - 修复版
用于替换 ToG 原有的 Wikidata/Freebase 查询接口
"""
from neo4j import GraphDatabase
from typing import List, Dict, Any, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Neo4jConnector:
    """Neo4j 数据库连接器"""

    def __init__(self, uri: str, username: str, password: str):
        """
        初始化 Neo4j 连接

        Args:
            uri: Neo4j 数据库地址,如 "bolt://localhost:7687"
            username: 用户名
            password: 密码
        """
        try:
            self.driver = GraphDatabase.driver(uri, auth=(username, password))
            self.driver.verify_connectivity()
            logger.info("成功连接到 Neo4j 数据库")
        except Exception as e:
            logger.error(f"连接 Neo4j 失败: {e}")
            raise

    def close(self):
        """关闭数据库连接"""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j 连接已关闭")

    def execute_query(self, cypher_query: str, parameters: Dict = None) -> List[Dict]:
        """
        执行 Cypher 查询

        Args:
            cypher_query: Cypher 查询语句
            parameters: 查询参数

        Returns:
            查询结果列表
        """
        try:
            with self.driver.session() as session:
                result = session.run(cypher_query, parameters or {})
                return [record.data() for record in result]
        except Exception as e:
            logger.error(f"Cypher查询失败: {e}")
            logger.error(f"查询语句: {cypher_query}")
            logger.error(f"参数: {parameters}")
            return []

    def search_entities_exact(self, entity_name: str) -> List[Dict]:
        """
        精确匹配实体
        """
        query = """
        MATCH (n)
        WHERE n.name = $entity_name 
        RETURN DISTINCT n.name AS entity_name
        LIMIT 10
        """
        params = {"entity_name": entity_name}
        results = self.execute_query(query, params)
        return [{"entity_name": str(item["entity_name"])} for item in results if item.get("entity_name")]

    def search_entities_partial(self, keyword: str) -> List[Dict]:
        """
        部分匹配实体（包含关键词）
        """
        query = """
        MATCH (n)
        WHERE n.name CONTAINS $keyword
        RETURN DISTINCT n.name AS entity_name
        LIMIT 10
        """
        params = {"keyword": keyword}
        results = self.execute_query(query, params)
        return [{"entity_name": str(item["entity_name"])} for item in results if item.get("entity_name")]

    def search_entities_fuzzy(self, keyword: str, limit: int = 10) -> List[Dict]:
        """
        模糊匹配实体
        """
        query = """
        MATCH (n)
        WHERE toLower(n.name) CONTAINS toLower($keyword)
        RETURN DISTINCT n.name AS entity_name
        LIMIT $limit
        """
        params = {"keyword": keyword, "limit": limit}
        results = self.execute_query(query, params)
        return [{"entity_name": str(item["entity_name"])} for item in results if item.get("entity_name")]

    def search_entities_containing(self, keyword: str, limit: int = 5) -> List[Dict]:
        """
        搜索包含特定关键词的实体
        """
        query = """
        MATCH (n)
        WHERE n.name CONTAINS $keyword
        RETURN DISTINCT n.name AS entity_name
        LIMIT $limit
        """
        params = {"keyword": keyword, "limit": limit}
        results = self.execute_query(query, params)
        return [{"entity_name": str(item["entity_name"])} for item in results if item.get("entity_name")]

    def search_entities(self, keyword: str, limit: int = 10) -> List[Dict]:
        """
        搜索实体 - 支持多种匹配策略
        """
        query = """
        MATCH (n)
        WHERE n.name IS NOT NULL 
          AND toLower(n.name) CONTAINS toLower($keyword)
        RETURN DISTINCT n.name AS entity_name
        LIMIT $limit
        """
        params = {"keyword": keyword, "limit": limit}
        results = self.execute_query(query, params)
        return [{"entity_name": str(item["entity_name"])} for item in results if item.get("entity_name")]

    def get_entity_neighbors(self, entity_name: str, depth: int = 1) -> List[Dict]:
        """
        获取实体的邻居节点(替代 ToG 的 get_neighbors 功能)

        Args:
            entity_name: 实体名称
            depth: 遍历深度

        Returns:
            邻居节点列表
        """
        query = """
        MATCH path = (n)-[r*1..{depth}]-(m)
        WHERE n.name = $entity_name
        RETURN DISTINCT 
            n.name as source_entity,
            type(r[0]) as relation,
            m.name as target_entity,
            labels(m) as target_labels,
            properties(m) as target_properties
        LIMIT 100
        """.replace("{depth}", str(depth))

        return self.execute_query(query, {"entity_name": entity_name})

    def get_relation_path(self, source: str, target: str, max_depth: int = 3) -> List[Dict]:
        """
        查找两个实体之间的关系路径

        Args:
            source: 源实体名称
            target: 目标实体名称
            max_depth: 最大路径长度

        Returns:
            路径列表
        """
        query = """
        MATCH path = shortestPath(
            (s)-[*1..{max_depth}]-(t)
        )
        WHERE s.name = $source AND t.name = $target
        RETURN 
            [node in nodes(path) | node.name] as node_names,
            [rel in relationships(path) | type(rel)] as relation_types,
            length(path) as path_length
        LIMIT 5
        """.replace("{max_depth}", str(max_depth))

        return self.execute_query(query, {"source": source, "target": target})

    def get_subgraph(self, entity_names: List[str], depth: int = 2) -> Dict[str, Any]:
        """
        获取多个实体周围的子图

        Args:
            entity_names: 实体名称列表
            depth: 子图深度

        Returns:
            子图数据(节点和边)
        """
        query = """
        MATCH path = (n)-[r*1..{depth}]-(m)
        WHERE n.name IN $entity_names
        UNWIND nodes(path) as node
        UNWIND relationships(path) as rel
        WITH DISTINCT node, rel
        RETURN 
            collect(DISTINCT {{
                id: id(node),
                name: node.name,
                labels: labels(node),
                properties: properties(node)
            }}) as nodes,
            collect(DISTINCT {{
                source: id(startNode(rel)),
                target: id(endNode(rel)),
                type: type(rel),
                properties: properties(rel)
            }}) as relationships
        """.replace("{depth}", str(depth))

        result = self.execute_query(query, {"entity_names": entity_names})
        return result[0] if result else {"nodes": [], "relationships": []}

    def execute_complex_query(self, cypher_query: str) -> List[Dict]:
        """
        执行复杂的自定义 Cypher 查询
        用于 LLM 生成的查询语句

        Args:
            cypher_query: 完整的 Cypher 查询语句

        Returns:
            查询结果
        """
        try:
            return self.execute_query(cypher_query)
        except Exception as e:
            logger.error(f"执行查询失败: {e}")
            logger.error(f"查询语句: {cypher_query}")
            return []

    def search_operations_by_keyword(self, keyword: str) -> List[Dict]:
        """
        专门搜索操作节点（Operation）
        用于处理 "系统激活" 这类查询
        """
        query = """
        MATCH (op:Operation)
        WHERE toLower(op.name) CONTAINS toLower($keyword)
        RETURN op.name AS entity_name
        LIMIT 10
        """
        params = {"keyword": keyword}
        results = self.execute_query(query, params)
        return [{"entity_name": str(item["entity_name"])} for item in results if item.get("entity_name")]

    def get_operation_steps(self, operation_name: str) -> List[Dict]:
        """
        获取某个操作的所有步骤
        """
        query = """
        MATCH (op:Operation {name: $operation_name})-[:HAS_STEP]->(step:Step)
        OPTIONAL MATCH path = (step)-[:NEXT_STEP*0..]->(nextStep:Step)
        WITH step, path
        ORDER BY length(path)
        RETURN DISTINCT step.name AS step_name
        """
        params = {"operation_name": operation_name}
        return self.execute_query(query, params)

    def get_operation_flow(self, operation_name: str) -> Dict[str, Any]:
        """
        获取操作的完整流程（包含步骤顺序）
        """
        query = """
        MATCH (op:Operation {name: $operation_name})-[:HAS_STEP]->(firstStep:Step)
        WHERE NOT (:Step)-[:NEXT_STEP]->(firstStep)
        MATCH path = (firstStep)-[:NEXT_STEP*0..]->(step:Step)
        WITH nodes(path) as steps
        RETURN [s in steps | s.name] as flow
        """
        params = {"operation_name": operation_name}
        result = self.execute_query(query, params)
        if result and result[0].get("flow"):
            return {
                "operation": operation_name,
                "steps": result[0]["flow"]
            }
        return {"operation": operation_name, "steps": []}

