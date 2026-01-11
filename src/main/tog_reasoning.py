import logging
from typing import List, Dict, Any, Union
import ollama
from neo4j_connector import Neo4jConnector

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class ToGReasoning:
    """
    ToG (Think-on-Graph) 推理引擎
    核心功能：基于图结构进行多跳推理，返回数据库中真实存在的答案实体
    """

    def __init__(
            self,
            neo4j_connector: Neo4jConnector,
            llm_model: str = "qwen3:8b",
            beam_width: int = 3,
            max_depth: int = 5
    ):
        self.neo4j = neo4j_connector
        self.llm_model = llm_model
        self.beam_width = beam_width
        self.max_depth = max_depth
        self.prompts = self._load_prompts()

        # 统计信息
        self.stats = {
            "total_queries": 0,
            "total_llm_calls": 0,
            "total_neo4j_queries": 0,
            "total_depth": 0
        }

    def _load_prompts(self) -> Dict[str, str]:
        """加载优化的提示词模板"""
        return {
            "relation_selection": """You are analyzing a knowledge graph to answer a question.

Question: {question}
Current entity: {entities}
Available relations from this entity: {relations}

Task: Select the {beam_width} most relevant relations that could lead to the answer.
Output format: relation1, relation2, relation3
Output only the relation names separated by commas, nothing else.

Selected relations:""",

            "entity_selection": """You are analyzing a knowledge graph to answer a question.

Question: {question}
Current path: ... --[{relation}]--> ?
Available target entities: {entities}

Task: Select the {beam_width} most relevant entities that could be or lead to the answer.
Output format: entity1, entity2, entity3
Output only the entity names separated by commas, nothing else.

Selected entities:""",

            "reasoning_check": """You are examining knowledge paths to determine if you can answer a question.

Question: {question}

Retrieved paths from knowledge graph:
{paths}

Task: Determine if these paths provide sufficient information to answer the question.
Output format: Yes or No (single word only)

Answer:""",

            "answer_extraction": """You are extracting the answer from knowledge graph paths.

Question: {question}

Knowledge paths retrieved:
{paths}

Task: Extract ONLY the entity name(s) that directly answer the question from the paths above.
The entities MUST appear in the paths shown above.

Rules:
1. Return ONLY entity names that appear in the paths
2. If multiple entities, separate with comma: entity1, entity2
3. NO explanations, NO extra text
4. Entity names MUST be exact matches from the paths

Answer entities:"""
        }

    def _call_llm(self, prompt: str, temperature: float = 0.0) -> str:
        """调用LLM"""
        try:
            self.stats["total_llm_calls"] += 1
            response = ollama.chat(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                options={
                    "temperature": temperature,
                    "num_predict": 2000
                }
            )
            return response['message']['content'].strip()
        except Exception as e:
            logger.error(f"LLM调用失败: {e}")
            return ""

    def predict(
            self,
            question: str,
            topic_entities: List[str]
    ) -> Union[str, List[str]]:
        """
        ToG推理核心接口

        Args:
            question: 问题文本
            topic_entities: 主题实体列表（问题中识别的实体）

        Returns:
            str 或 List[str]: 预测的答案实体（数据库中真实存在的实体）
        """
        try:
            self.stats["total_queries"] += 1

            if not topic_entities:
                return ""

            # 迭代beam search探索
            current_paths = []
            for depth in range(self.max_depth):
                current_paths = self._beam_search_step(
                    current_paths, question, topic_entities
                )

                if not current_paths:
                    break

                self.stats["total_depth"] = depth + 1

                # 判断是否可以回答
                if self._can_answer(question, current_paths):
                    break

            # 从路径中提取答案实体
            if current_paths:
                return self._extract_answer(question, current_paths)
            else:
                return ""

        except Exception as e:
            logger.error(f"推理失败: {e}", exc_info=True)
            return ""

    def _beam_search_step(
            self,
            current_paths: List[List[Dict]],
            question: str,
            topic_entities: List[str]
    ) -> List[List[Dict]]:
        """执行一步beam search"""
        # 初始化：第一步从主题实体开始
        if not current_paths:
            current_paths = [[{
                "source": None,
                "relation": None,
                "target": entity
            }] for entity in topic_entities]

        # 获取当前路径的尾实体
        tail_entities = [path[-1]["target"] for path in current_paths if path]

        # 探索关系
        entity_relations = self._select_relations(tail_entities, question)

        # 探索目标实体
        exploration_results = self._select_entities(entity_relations, question)

        # 扩展路径
        new_paths = []
        for path in current_paths:
            tail = path[-1]["target"]
            for result in exploration_results:
                if result["source"] == tail:
                    for target in result["targets"]:
                        new_path = path + [{
                            "source": result["source"],
                            "relation": result["relation"],
                            "target": target
                        }]
                        new_paths.append(new_path)

        # Beam pruning: 保留top-k路径
        return new_paths[:self.beam_width] if new_paths else current_paths

    def _select_relations(
            self,
            entities: List[str],
            question: str
    ) -> Dict[str, List[str]]:
        """为每个实体选择最相关的关系"""
        entity_relations = {}

        for entity in entities:
            # 从图数据库获取邻居
            self.stats["total_neo4j_queries"] += 1
            neighbors = self.neo4j.get_entity_neighbors(entity, depth=1)

            # 提取所有关系类型
            relations = list(set([
                n.get("relation") for n in neighbors if n.get("relation")
            ]))

            if not relations:
                entity_relations[entity] = []
                continue

            # LLM选择最相关的关系
            if len(relations) <= self.beam_width:
                selected = relations
            else:
                prompt = self.prompts["relation_selection"].format(
                    question=question,
                    entities=entity,
                    relations=", ".join(relations),
                    beam_width=self.beam_width
                )
                response = self._call_llm(prompt)
                selected = [r.strip() for r in response.split(",") if r.strip()]
                selected = selected[:self.beam_width]

            entity_relations[entity] = selected

        return entity_relations

    def _select_entities(
            self,
            entity_relations: Dict[str, List[str]],
            question: str
    ) -> List[Dict[str, Any]]:
        """为每个(实体,关系)对选择最相关的目标实体"""
        results = []

        for source, relations in entity_relations.items():
            for relation in relations:
                # 获取邻居
                self.stats["total_neo4j_queries"] += 1
                neighbors = self.neo4j.get_entity_neighbors(source, depth=1)

                # 过滤该关系的目标实体
                targets = list(dict.fromkeys([
                    n.get("target_entity")
                    for n in neighbors
                    if n.get("relation") == relation and n.get("target_entity")
                ]))

                if not targets:
                    continue

                # LLM选择最相关的目标实体
                if len(targets) <= self.beam_width:
                    selected = targets
                else:
                    prompt = self.prompts["entity_selection"].format(
                        question=question,
                        relation=relation,
                        entities=", ".join(targets[:30]),  # 限制提示词长度
                        beam_width=self.beam_width
                    )
                    response = self._call_llm(prompt)
                    selected = [e.strip() for e in response.split(",") if e.strip()]
                    selected = selected[:self.beam_width]

                results.append({
                    "source": source,
                    "relation": relation,
                    "targets": selected
                })

        return results

    def _can_answer(self, question: str, paths: List[List[Dict]]) -> bool:
        """判断当前路径是否足以回答问题"""
        if not paths or not any(paths):
            return False

        paths_text = self._format_paths(paths)
        prompt = self.prompts["reasoning_check"].format(
            question=question,
            paths=paths_text
        )

        response = self._call_llm(prompt).lower()
        return "yes" in response

    def _extract_answer(self, question: str, paths: List[List[Dict]]) -> Union[str, List[str]]:
        """从路径中提取答案实体"""
        paths_text = self._format_paths(paths)
        prompt = self.prompts["answer_extraction"].format(
            question=question,
            paths=paths_text
        )

        answer = self._call_llm(prompt, temperature=0.0)

        # 清理答案
        answer = answer.strip().strip('"').strip("'")

        # 移除编号和多余格式
        import re
        answer = re.sub(r'^\d+[\.\)]\s*', '', answer)
        answer = re.sub(r'^[-•]\s*', '', answer)

        # 处理多行
        lines = [line.strip() for line in answer.split('\n') if line.strip()]

        # 从paths中提取所有可能的实体名（用于验证）
        all_entities = set()
        for path in paths:
            for step in path:
                if step.get("target"):
                    all_entities.add(step["target"])
                if step.get("source"):
                    all_entities.add(step["source"])

        # 验证并过滤答案
        valid_entities = []
        if len(lines) > 1:
            # 多个答案实体
            for line in lines:
                line = re.sub(r'^\d+[\.\)]\s*', '', line)
                if line in all_entities:
                    valid_entities.append(line)
        elif len(lines) == 1:
            # 单行可能包含逗号分隔的多个实体
            if ',' in lines[0]:
                candidates = [e.strip() for e in lines[0].split(',')]
                valid_entities = [e for e in candidates if e in all_entities]
            else:
                if lines[0] in all_entities:
                    valid_entities.append(lines[0])

        # 返回结果
        if len(valid_entities) > 1:
            return valid_entities
        elif len(valid_entities) == 1:
            return valid_entities[0]
        else:
            # 如果验证失败，返回原始答案（降级处理）
            return answer if answer else ""

    def _format_paths(self, paths: List[List[Dict]]) -> str:
        """格式化路径用于LLM阅读"""
        formatted = []
        for i, path in enumerate(paths, 1):
            steps = []
            for step in path:
                if step['relation'] is not None:
                    steps.append(f"{step['source']} --[{step['relation']}]--> {step['target']}")
            if steps:
                formatted.append(f"Path {i}: {' -> '.join(steps)}")
        return "\n".join(formatted) if formatted else "No paths found"

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = self.stats.copy()
        if stats["total_queries"] > 0:
            stats["avg_depth"] = stats["total_depth"] / stats["total_queries"]
            stats["avg_llm_calls_per_query"] = stats["total_llm_calls"] / stats["total_queries"]
            stats["avg_neo4j_queries_per_query"] = stats["total_neo4j_queries"] / stats["total_queries"]
        return stats

    def print_stats(self):
        """打印统计信息"""
        stats = self.get_stats()
        print("-" * 80)
        print("ToG 推理引擎统计:")
        print(f"  总查询数:          {stats['total_queries']:>8}")
        print(f"  LLM 调用次数:      {stats['total_llm_calls']:>8}")
        print(f"  Neo4j 查询次数:    {stats['total_neo4j_queries']:>8}")
        if stats['total_queries'] > 0:
            print(f"  平均探索深度:      {stats['avg_depth']:>8.2f}")
            print(f"  平均 LLM 调用:    {stats['avg_llm_calls_per_query']:>8.2f} calls/query")
            print(f"  平均 Neo4j 查询:  {stats['avg_neo4j_queries_per_query']:>8.2f} queries/query")
        print("-" * 80)