# extract_relation_prompt = """Please provide as few highly relevant relations as possible to the question and its subobjectives from the following relations (separated by semicolons).
# Here is an example:
# Q: Name the president of the country whose main spoken language was Brahui in 1980?
# Subobjectives: ['Identify the countries where the main spoken language is Brahui', 'Find the president of each country', 'Determine the president from 1980']
# Topic Entity: Brahui Language
# Relations: language.human_language.main_country; language.human_language.language_family; language.human_language.iso_639_3_code; base.rosetta.languoid.parent; language.human_language.writing_system; base.rosetta.languoid.languoid_class; language.human_language.countries_spoken_in; kg.object_profile.prominent_type; base.rosetta.languoid.document; base.ontologies.ontology_instance.equivalent_instances; base.rosetta.languoid.local_name; language.human_language.region
# The output is:
# ['language.human_language.main_country','language.human_language.countries_spoken_in','base.rosetta.languoid.parent']
#
# Now you need to directly output relations highly related to the following question and its subobjectives in list format without other information or notes.
# Q: """


extract_relation_prompt= """Role: 你是一个运维知识图谱专家，擅长从复杂的 IT 拓扑和指标关系中提取核心关联路径。

Bilingual Requirement (双语处理要求): > 重点注意：由于运维知识库通常包含中英文混搭（如：中文描述与英文指标名/错误码），在筛选关系时，请执行跨语言语义对齐。例如，中文的“内存溢出”与英文关系 system.metric.oom_score 或 memory.usage.exception 具有高度相关性，请基于语义而非字面匹配进行筛选。

Task: 请从给定的关系列表中，筛选出与问题及其子目标最相关的极少数核心关系（Relations）。

Example:

Q: 交易响应时间（RT）突增是由哪个容器的 CPU 限制导致的？

Topic Entity: 交易 (Transaction)

Relations: service.endpoint.path; infrastructure.container.cpu_limit; metrics.response_time.p99; common.name.cn; deployment.strategy; node.hardware.spec; relationship.hosted_on; alert.severity.level; app.runtime.jvm_heap

The output is: ['service.endpoint.path', 'relationship.hosted_on', 'infrastructure.container.cpu_limit']

Real Input

Now you need to directly output relations in the given 'Relations' list below which highly related to the following question in list format without other information or notes.
You must only select one or more relations from the provided 'Relations' list; do not generate any relations that are not present in the list.

Q: """

system_prompt = "You are an AI assistant that helps people find information."


prune_entity_prompt = """
Which entities in the following list ([] in Triples) can be used to answer question? Please provide the minimum possible number of entities, and strictly adhering to the constraints mentioned in the question. Remember at least keep one entity!
Here is an example:
Q: The movie featured Miley Cyrus and was produced by Tobin Armbrust?
Triplets: Tobin Armbrust film.producer.film ['The Resident', 'So Undercover', 'Let Me In', 'Begin Again', 'The Quiet Ones', 'A Walk Among the Tombstones']
Output: ['So Undercover']

Now you need to directly output the entities from [] in Triplets for the following question in list format without other information or notes.
Q: """

# utils.py 内部定义

# answer_prompt = """Given a question and the associated retrieved knowledge graph triplets (entity, relation, entity), your task is to answer the question.
#
# ### Instructions:
# 1. **Priority**: Use the provided Knowledge Triplets to find the answer.
# 2. **Supplement**: If the triplets are insufficient, use your internal knowledge to complete the answer.
# 3. **Format**: You MUST output the answer as a Python-style list of entity names, for example: ["Entity Name 1", "Entity Name 2"].
# 4. **Constraint**: Provide at most 3 most relevant entity names. If you cannot find a certain answer, provide your best guess based on the context.
#
# ### Context:
# - Topic Entities: {topic_entities}
# - Knowledge Triplets:
# {triplets}
#
# ### Question:
# {question}
#
# ### Answer (List format only):"""

answer_prompt = """给定一个运维场景下的问题及相关的知识图谱三元组 (实体, 关系, 实体)，请识别出导致故障的【根因实体】。

### 任务指令：
1. **优先权**：优先根据提供的知识三元组（Triplets）分析故障传播路径。
2. **推理逻辑**：运维场景中，异常通常沿“基础组件 -> 中间件 -> 应用服务”传播，请寻找链路最上游的故障点。
3. **输出格式**：你必须仅输出最终确定的一个根因实体名称。如果无法确定，请输出 None。严禁输出列表或多余解释。
4. **数量限制**：仅输出 1 个最可能的实体。

### 示例 (Example)：
- Topic Entities: ["订单服务 (Order Service)"]
- Knowledge Triplets: 
("订单服务", "依赖", "MySQL数据库")
("MySQL数据库", "状态", "CPU使用率高")
("MySQL数据库", "触发", "慢查询告警")
- Question: 订单服务出现大量接口超时 (504 Gateway Timeout)，可能的原因是什么？
- Answer: MySQL数据库

### 真实数据 (Real Data)：
- Topic Entities: {topic_entities}
- Knowledge Triplets: 
{triplets}

### 问题 (Question)： 
{question}

### 最终答案 (仅输出实体名或 None):"""
