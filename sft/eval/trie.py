      
"""
SID Trie 实现

高效的 Trie 树数据结构，用于：
1. 存储有效的 SID 序列
2. 支持受限解码时获取有效后续 token
3. 集成 Hugging Face LogitsProcessor
"""

import json
import pickle
from typing import Any, Dict, List, Optional, Set, Tuple


class TrieNode:
    """
    Trie 树节点
    
    每个节点代表一个 SID 层级中的 token。
    """
    __slots__ = ['children', 'is_end', 'item_id', 'count']
    
    def __init__(self) -> None:
        self.children: Dict[int, "TrieNode"] = {}
        self.is_end: bool = False  # 是否为完整 SID 的终止节点
        self.item_id: Optional[str] = None  # 关联的资源 ID
        self.count: int = 0  # 经过此节点的 SID 数量


class SIDTrie:
    """
    SID Trie 树
    
    功能:
        1. 存储所有有效的 SID 序列
        2. 快速查询给定前缀的有效后续 token
        3. 验证 SID 序列是否有效
        4. 支持序列化/反序列化
    
    SID 格式:
        - 每个 SID 是多层级量化索引的序列
        - 例如: [12, 45, 0, 123] 对应 "<a_12><b_45><c_0><d_123>"
    """
    
    def __init__(
        self, 
        num_levels: int = 4,
        codebook_sizes: Optional[List[int]] = None,
        level_prefixes: Optional[List[str]] = None
    ) -> None:
        """
        初始化 Trie
        
        Args:
            num_levels: SID 层级数量
            codebook_sizes: 各层码本大小，用于验证 token 范围
            level_prefixes: 各层级前缀字符，如 ["a", "b", "c", "d"]
        """
        self.root = TrieNode()
        self.num_levels = num_levels
        self.codebook_sizes = codebook_sizes or [256] * num_levels
        self.level_prefixes = level_prefixes or [chr(ord('a') + i) for i in range(num_levels)]
        self._num_sids = 0
    
    @property
    def num_sids(self) -> int:
        """返回 Trie 中的 SID 总数"""
        return self._num_sids
    
    def insert(self, sid_tokens: List[int], item_id: Optional[str] = None) -> bool:
        """
        插入一个 SID 序列到 Trie
        
        Args:
            sid_tokens: SID token 索引列表，如 [12, 45, 0, 123]
            item_id: 可选的关联资源 ID
        
        Returns:
            bool: 是否为新插入（非重复）
        """
        if len(sid_tokens) != self.num_levels:
            raise ValueError(
                f"SID 长度 {len(sid_tokens)} 与预期层级数 {self.num_levels} 不匹配"
            )
        
        # 验证 token 范围
        for level, token in enumerate(sid_tokens):
            if not (0 <= token < self.codebook_sizes[level]):
                raise ValueError(
                    f"层级 {level} 的 token {token} 超出范围 [0, {self.codebook_sizes[level]})"
                )
        
        node = self.root
        is_new = False
        
        for token in sid_tokens:
            if token not in node.children:
                node.children[token] = TrieNode()
                is_new = True
            node = node.children[token]
            node.count += 1
        
        if not node.is_end:
            node.is_end = True
            self._num_sids += 1
            is_new = True
        
        if item_id is not None:
            node.item_id = item_id
        
        return is_new
    
    def search(self, sid_tokens: List[int]) -> bool:
        """
        检查完整 SID 序列是否存在
        
        Args:
            sid_tokens: SID token 索引列表
        
        Returns:
            bool: SID 是否存在于 Trie 中
        """
        node = self._traverse(sid_tokens)
        return node is not None and node.is_end
    
    def is_valid_prefix(self, prefix_tokens: List[int]) -> bool:
        """
        检查前缀是否有效（存在于 Trie 中）
        
        Args:
            prefix_tokens: 前缀 token 列表
        
        Returns:
            bool: 前缀是否有效
        """
        return self._traverse(prefix_tokens) is not None
    
    def get_valid_next_tokens(self, prefix_tokens: List[int]) -> List[int]:
        """
        获取给定前缀的所有有效后续 token
        
        这是受限解码的核心方法。
        
        Args:
            prefix_tokens: 当前 SID 前缀的 token 列表
        
        Returns:
            List[int]: 有效的后续 token 索引列表
        """
        if len(prefix_tokens) >= self.num_levels:
            return []  # 已达到最大层级
        
        node = self._traverse(prefix_tokens)
        if node is None:
            return []
        
        return list(node.children.keys())
    
    def get_item_id(self, sid_tokens: List[int]) -> Optional[str]:
        """
        获取 SID 关联的资源 ID
        
        Args:
            sid_tokens: SID token 列表
        
        Returns:
            Optional[str]: 资源 ID，不存在则返回 None
        """
        node = self._traverse(sid_tokens)
        if node is not None and node.is_end:
            return node.item_id
        return None
    
    def _traverse(self, tokens: List[int]) -> Optional[TrieNode]:
        """
        遍历 Trie 到指定 token 序列的节点
        
        Args:
            tokens: token 序列
        
        Returns:
            Optional[TrieNode]: 目标节点，不存在则返回 None
        """
        node = self.root
        for token in tokens:
            if token not in node.children:
                return None
            node = node.children[token]
        return node
    
    def tokens_to_sid_str(self, sid_tokens: List[int]) -> str:
        """
        将 token 列表转换为 SID 字符串
        
        Args:
            sid_tokens: token 索引列表，如 [12, 45, 0, 123]
        
        Returns:
            str: SID 字符串，如 "<a_12><b_45><c_0><d_123>"
        """
        parts = []
        for level, token in enumerate(sid_tokens):
            prefix = self.level_prefixes[level] if level < len(self.level_prefixes) else chr(ord('a') + level)
            parts.append(f"<{prefix}_{token}>")
        return "".join(parts)
    
    def sid_str_to_tokens(self, sid_str: str) -> List[int]:
        """
        将 SID 字符串解析为 token 列表
        
        Args:
            sid_str: SID 字符串，如 "<a_12><b_45><c_0><d_123>"
        
        Returns:
            List[int]: token 索引列表
        """
        import re
        pattern = r"<([a-z])_(\d+)>"
        matches = re.findall(pattern, sid_str)
        return [int(idx) for _, idx in matches]
    
    def get_all_sids(self) -> List[List[int]]:
        """
        获取 Trie 中所有完整的 SID 序列
        
        Returns:
            List[List[int]]: 所有 SID token 列表
        """
        sids: List[List[int]] = []
        self._collect_sids(self.root, [], sids)
        return sids
    
    def _collect_sids(
        self, 
        node: TrieNode, 
        current: List[int], 
        result: List[List[int]]
    ) -> None:
        """递归收集所有 SID"""
        if node.is_end:
            result.append(current.copy())
        
        for token, child in node.children.items():
            current.append(token)
            self._collect_sids(child, current, result)
            current.pop()
    
    # ==========================================================================
    # 序列化/反序列化
    # ==========================================================================
    
    def to_dict(self) -> Dict[str, Any]:
        """
        将 Trie 序列化为字典
        
        Returns:
            Dict: 可 JSON 序列化的字典
        """
        return {
            "num_levels": self.num_levels,
            "codebook_sizes": self.codebook_sizes,
            "level_prefixes": self.level_prefixes,
            "num_sids": self._num_sids,
            "sids": [
                {
                    "tokens": sid,
                    "item_id": self.get_item_id(sid)
                }
                for sid in self.get_all_sids()
            ]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SIDTrie":
        """
        从字典反序列化 Trie
        
        Args:
            data: 序列化的 Trie 数据
        
        Returns:
            SIDTrie: 重建的 Trie 实例
        """
        trie = cls(
            num_levels=data["num_levels"],
            codebook_sizes=data["codebook_sizes"],
            level_prefixes=data.get("level_prefixes")
        )
        
        for sid_data in data.get("sids", []):
            tokens = sid_data["tokens"]
            item_id = sid_data.get("item_id")
            trie.insert(tokens, item_id)
        
        return trie
    
    def save(self, path: str) -> None:
        """
        保存 Trie 到文件
        
        Args:
            path: 文件路径（支持 .json 或 .pkl）
        """
        if path.endswith(".json"):
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        else:
            with open(path, "wb") as f:
                pickle.dump(self, f)
    
    @classmethod
    def load(cls, path: str) -> "SIDTrie":
        """
        从文件加载 Trie
        
        Args:
            path: 文件路径
        
        Returns:
            SIDTrie: 加载的 Trie 实例
        """
        if path.endswith(".json"):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return cls.from_dict(data)
        else:
            with open(path, "rb") as f:
                return pickle.load(f)
    
    def estimate_memory_bytes(self) -> int:
        """
        估算 Trie 占用的内存大小
        
        Returns:
            int: 估计的内存字节数
        """
        # 粗略估算：每个节点约 200 字节
        node_count = self._count_nodes(self.root)
        return node_count * 200
    
    def _count_nodes(self, node: TrieNode) -> int:
        """递归计算节点数量"""
        count = 1
        for child in node.children.values():
            count += self._count_nodes(child)
        return count


# ==============================================================================
# 批量构建 Trie
# ==============================================================================

def build_trie_from_index_file(
    index_path: str,
    num_levels: int = 4,
    codebook_sizes: Optional[List[int]] = None
) -> SIDTrie:
    """
    从索引文件构建 Trie
    
    索引文件格式（JSON）:
        {
            "0": ["<a_12>", "<b_45>", "<c_0>", "<d_123>"],
            "1": ["<a_0>", "<b_1>", "<c_2>", "<d_3>"],
            ...
        }
    
    Args:
        index_path: 索引文件路径
        num_levels: SID 层级数
        codebook_sizes: 各层码本大小
    
    Returns:
        SIDTrie: 构建的 Trie
    """
    import re
    
    with open(index_path, "r", encoding="utf-8") as f:
        index_data = json.load(f)
    
    trie = SIDTrie(num_levels=num_levels, codebook_sizes=codebook_sizes)
    pattern = r"<([a-z])_(\d+)>"
    
    for item_id, sid_parts in index_data.items():
        tokens = []
        for part in sid_parts:
            match = re.match(pattern, part)
            if match:
                tokens.append(int(match.group(2)))
        
        if len(tokens) == num_levels:
            trie.insert(tokens, item_id=str(item_id))
    
    return trie

    