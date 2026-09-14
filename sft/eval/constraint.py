      
"""
LLM Service - 受限解码约束

实现 Trie 约束的 LogitsProcessor，用于生成有效的 SID 序列。
"""

import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
from transformers import LogitsProcessor

# 本地导入
from trie import SIDTrie

logger = logging.getLogger(__name__)


class SIDConstrainedLogitsProcessor(LogitsProcessor):
    """
    SID 受限解码 LogitsProcessor
    
    在 LLM 生成过程中，根据 Trie 约束只允许生成有效的 SID token。
    
    工作原理:
        1. 解析当前已生成的 SID token 序列
        2. 查询 Trie 获取有效的后续 token
        3. 将无效 token 的 logits 设为负无穷
    """
    
    def __init__(
        self,
        trie: SIDTrie,
        tokenizer: Any,
        sid_token_pattern: str = r"<([a-z])_(\d+)>",
        force_constraint: bool = True,
        eos_token_id: Optional[int] = None,
        sep_token_id: Optional[int] = None
    ) -> None:
        """
        初始化 LogitsProcessor
        
        Args:
            trie: SIDTrie 实例
            tokenizer: Tokenizer 实例
            sid_token_pattern: SID token 的正则模式
            force_constraint: 是否强制约束（False 时仅做软约束）
            eos_token_id: 结束符 ID
            sep_token_id: 分隔符 ID (用于多 SID 生成, 如逗号)
        """
        self.trie = trie
        self.tokenizer = tokenizer
        self.sid_token_pattern = re.compile(sid_token_pattern)
        self.force_constraint = force_constraint
        self.eos_token_id = eos_token_id
        self.sep_token_id = sep_token_id
        
        # 预构建 SID token 到 token_id 的映射
        self._sid_token_to_id: Dict[str, int] = {}
        self._id_to_sid_token: Dict[int, str] = {}
        self._level_token_ids: Dict[int, Set[int]] = {}
        
        self._build_token_mapping()
    
    def _build_token_mapping(self) -> None:
        # Code remains same...
        vocab = self.tokenizer.get_vocab()
        
        for token_str, token_id in vocab.items():
            match = self.sid_token_pattern.match(token_str)
            if match:
                level_char = match.group(1)
                level = ord(level_char) - ord('a')
                
                self._sid_token_to_id[token_str] = token_id
                self._id_to_sid_token[token_id] = token_str
                
                if level not in self._level_token_ids:
                    self._level_token_ids[level] = set()
                self._level_token_ids[level].add(token_id)
        
        logger.info(f"构建 SID token 映射完成，共 {len(self._sid_token_to_id)} 个 token")
        # debug logs...

    def _parse_generated_sid_tokens(
        self,
        generated_ids: torch.Tensor
    ) -> List[int]:
        """
        解析 *当前正在生成* 的 SID token 序列
        如果遇到分隔符，则重置序列，只返回分隔符之后的部分。
        """
        sid_tokens = []
        token_ids = generated_ids.tolist()

        # 找到最后一个分隔符的位置
        last_sep_idx = -1
        if self.sep_token_id is not None:
            # 倒序查找
            for i in range(len(token_ids) - 1, -1, -1):
                if token_ids[i] == self.sep_token_id:
                    last_sep_idx = i
                    break

        # 只解析最后一个分隔符之后的 tokens
        relevant_token_ids = token_ids[last_sep_idx + 1:]

        for token_id in relevant_token_ids:
            token_id = int(token_id)
            if token_id in self._id_to_sid_token:
                token_str = self._id_to_sid_token[token_id]
                match = self.sid_token_pattern.match(token_str)
                if match:
                    index = int(match.group(2))
                    sid_tokens.append(index)

        return sid_tokens

    def _parse_all_sids_from_ids(self, token_ids: List[int]) -> List[List[int]]:
        """
        从 token IDs 解析所有 SID 序列（支持多个 SID，以 sep_token 分隔）

        Args:
            token_ids: 完整的 token ID 列表

        Returns:
            List[List[int]]: 所有 SID 序列的列表
        """
        all_sids = []
        current_sid = []

        for token_id in token_ids:
            if self.sep_token_id is not None and token_id == self.sep_token_id:
                # 遇到分隔符，保存当前 SID 并开始新的 SID
                if current_sid:
                    all_sids.append(current_sid)
                    current_sid = []
            elif token_id in self._id_to_sid_token:
                token_str = self._id_to_sid_token[token_id]
                match = self.sid_token_pattern.match(token_str)
                if match:
                    index = int(match.group(2))
                    current_sid.append(index)

        # 添加最后一个 SID（如果有）
        if current_sid:
            all_sids.append(current_sid)

        return all_sids
    
    def _get_current_level(self, sid_tokens: List[int]) -> int:
        return len(sid_tokens)
    
    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        batch_size = input_ids.shape[0]
        vocab_size = scores.shape[1]
        device = scores.device

        for batch_idx in range(batch_size):
            current_sid_tokens = self._parse_generated_sid_tokens(input_ids[batch_idx])
            current_level = self._get_current_level(current_sid_tokens)

            # 如果已完成当前 SID 的所有层级
            if current_level >= self.trie.num_levels:
                if self.force_constraint:
                    mask = torch.full((vocab_size,), float("-inf"), device=device)
                    # 允许 EOS (结束全部)
                    if self.eos_token_id is not None:
                        mask[self.eos_token_id] = 0
                    # 允许 SEP (开始下一个)
                    if self.sep_token_id is not None:
                        mask[self.sep_token_id] = 0

                    scores[batch_idx] = scores[batch_idx] + mask
                continue

            # 常规约束逻辑: 获取下一层有效 token
            valid_next_indices = self.trie.get_valid_next_tokens(current_sid_tokens)

            if not valid_next_indices:
                continue # 无有效后续，可能出错了

            valid_token_ids = set()
            for idx in valid_next_indices:
                level_prefix = self.trie.level_prefixes[current_level]
                sid_token_str = f"<{level_prefix}_{idx}>"
                if sid_token_str in self._sid_token_to_id:
                    valid_token_ids.add(self._sid_token_to_id[sid_token_str])

            if self.force_constraint:
                mask = torch.full((vocab_size,), float("-inf"), device=device)
                for token_id in valid_token_ids:
                    if 0 <= token_id < vocab_size:
                        mask[token_id] = 0
                scores[batch_idx] = scores[batch_idx] + mask

        return scores

    def parse_sids_from_text(self, text: str) -> List[List[int]]:
        """
        从文本中解析所有 SID 序列（支持多个 SID，以回车分隔）

        Args:
            text: 包含 SID 的文本

        Returns:
            List[List[int]]: 所有 SID 序列的列表
        """
        all_sids = []

        # 按回车分割
        lines = text.split('\n')

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # 解析单行中的所有 SID
            tokens = self.trie.sid_str_to_tokens(line)
            if tokens:
                all_sids.append(tokens)

        return all_sids
    
    def update_trie(self, new_trie: SIDTrie) -> None:
        """
        更新 Trie（热更新）

        Args:
            new_trie: 新的 SIDTrie 实例
        """
        self.trie = new_trie
        logger.info(f"LogitsProcessor Trie 已更新，共 {new_trie.num_sids} 个 SID")


class CachedSIDConstrainedLogitsProcessor(SIDConstrainedLogitsProcessor):
    """
    带缓存优化的 SIDConstrainedLogitsProcessor

    优化点:
        1. 缓存 Trie 查询结果 (前缀 -> 有效token列表)
        2. 缓存 token_id 转换结果
        3. 提供缓存命中率统计
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # 缓存：前缀(tuple) -> 有效token索引列表
        self._trie_cache: Dict[Tuple[int, ...], List[int]] = {}

        # 缓存：token索引 -> token_id (避免重复字符串拼接)
        self._token_id_cache: Dict[Tuple[int, int], int] = {}

        # 预计算所有可能的 token 映射 (层级, 索引) -> token_id
        self._precompute_token_mappings()

        # 统计信息
        self._cache_hits = 0
        self._cache_misses = 0
        self._total_calls = 0

    def _precompute_token_mappings(self):
        """预计算所有层级/token到token_id的映射"""
        self._precomputed_mappings: Dict[Tuple[int, int], int] = {}
        for level in range(self.trie.num_levels):
            level_prefix = self.trie.level_prefixes[level]
            codebook_size = self.trie.codebook_sizes[level]
            for idx in range(codebook_size):
                sid_token_str = f"<{level_prefix}_{idx}>"
                if sid_token_str in self._sid_token_to_id:
                    self._precomputed_mappings[(level, idx)] = self._sid_token_to_id[sid_token_str]
        logger.info(f"预计算 token 映射完成: {len(self._precomputed_mappings)} 个")

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        self._total_calls += 1
        batch_size = input_ids.shape[0]
        vocab_size = scores.shape[1]
        device = scores.device

        for batch_idx in range(batch_size):
            current_sid_tokens = self._parse_generated_sid_tokens(input_ids[batch_idx])
            current_level = self._get_current_level(current_sid_tokens)

            # 如果已完成当前 SID 的所有层级
            if current_level >= self.trie.num_levels:
                if self.force_constraint:
                    mask = torch.full((vocab_size,), float("-inf"), device=device)
                    if self.eos_token_id is not None:
                        mask[self.eos_token_id] = 0
                    if self.sep_token_id is not None:
                        mask[self.sep_token_id] = 0
                    scores[batch_idx] = scores[batch_idx] + mask
                continue

            # 使用缓存的 Trie 查询
            cache_key = tuple(current_sid_tokens)
            if cache_key in self._trie_cache:
                valid_next_indices = self._trie_cache[cache_key]
                self._cache_hits += 1
            else:
                valid_next_indices = self.trie.get_valid_next_tokens(current_sid_tokens)
                self._trie_cache[cache_key] = valid_next_indices
                self._cache_misses += 1

            if not valid_next_indices:
                continue

            # 使用预计算的 token_id 映射
            valid_token_ids = []
            for idx in valid_next_indices:
                token_id = self._precomputed_mappings.get((current_level, idx))
                if token_id is not None:
                    valid_token_ids.append(token_id)

            if self.force_constraint:
                mask = torch.full((vocab_size,), float("-inf"), device=device)
                for token_id in valid_token_ids:
                    if 0 <= token_id < vocab_size:
                        mask[token_id] = 0
                scores[batch_idx] = scores[batch_idx] + mask

        return scores

    def get_cache_stats(self) -> Dict[str, float]:
        """获取缓存统计信息"""
        total = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total if total > 0 else 0.0
        return {
            "total_calls": self._total_calls,
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate": hit_rate,
            "cache_size": len(self._trie_cache),
        }

    def reset_stats(self):
        """重置统计信息"""
        self._cache_hits = 0
        self._cache_misses = 0
        self._total_calls = 0

    def clear_cache(self):
        """清空缓存"""
        self._trie_cache.clear()
        self.reset_stats()
        logger.info("缓存已清空")


class SimpleSIDLogitsProcessor(LogitsProcessor):
    """
    简化版 SID LogitsProcessor
    
    假设 SID token 在词表中连续排列，使用偏移量计算。
    适用于专门为 SID 设计的词表。
    """
    
    def __init__(
        self,
        trie: SIDTrie,
        sid_token_offset: int,
        level_sizes: List[int],
        eos_token_id: Optional[int] = None
    ) -> None:
        """
        初始化
        
        Args:
            trie: SIDTrie 实例
            sid_token_offset: SID token 在词表中的起始偏移
            level_sizes: 各层级的 token 数量
            eos_token_id: EOS token ID
        """
        self.trie = trie
        self.sid_token_offset = sid_token_offset
        self.level_sizes = level_sizes
        self.eos_token_id = eos_token_id
        
        # 计算各层级的 token 范围
        self.level_ranges: List[Tuple[int, int]] = []
        offset = sid_token_offset
        for size in level_sizes:
            self.level_ranges.append((offset, offset + size))
            offset += size
    
    def _get_level_and_index(self, token_id: int) -> Optional[Tuple[int, int]]:
        """
        根据 token_id 获取层级和索引
        
        Returns:
            Optional[Tuple[int, int]]: (level, index) 或 None
        """
        for level, (start, end) in enumerate(self.level_ranges):
            if start <= token_id < end:
                return level, token_id - start
        return None
    
    def _parse_sid_from_ids(self, token_ids: torch.Tensor) -> List[int]:
        """从 token IDs 解析 SID"""
        sid_tokens = []
        for token_id in token_ids.tolist():
            result = self._get_level_and_index(int(token_id))
            if result is not None:
                _, index = result
                sid_tokens.append(index)
        return sid_tokens
    
    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        """应用约束"""
        batch_size = input_ids.shape[0]
        vocab_size = scores.shape[1]
        device = scores.device
        
        for batch_idx in range(batch_size):
            current_sid = self._parse_sid_from_ids(input_ids[batch_idx])
            current_level = len(current_sid)
            
            if current_level >= len(self.level_ranges):
                # 已完成，只允许 EOS
                mask = torch.full((vocab_size,), float("-inf"), device=device)
                if self.eos_token_id is not None:
                    mask[self.eos_token_id] = 0
                scores[batch_idx] = scores[batch_idx] + mask
                continue
            
            # 获取有效后续
            valid_indices = self.trie.get_valid_next_tokens(current_sid)
            
            if valid_indices:
                start, _ = self.level_ranges[current_level]
                valid_token_ids = [start + idx for idx in valid_indices]
                
                mask = torch.full((vocab_size,), float("-inf"), device=device)
                for token_id in valid_token_ids:
                    if 0 <= token_id < vocab_size:
                        mask[token_id] = 0
                
                scores[batch_idx] = scores[batch_idx] + mask
        
        return scores
    
    def update_trie(self, new_trie: SIDTrie) -> None:
        """
        更新 Trie（热更新）

        Args:
            new_trie: 新的 SIDTrie 实例
        """
        self.trie = new_trie
        logger.info(f"SimpleSIDLogitsProcessor Trie 已更新")


# ==============================================================================
# Level 1: FlatTrieProcessor - 展平 Trie 结构优化
# ==============================================================================

class FlatTrieProcessor(SIDConstrainedLogitsProcessor):
    """
    Level 1 优化: 展平 Trie 结构，使用数组索引替代字典遍历

    优化点:
        1. 将 Trie 树展平为连续数组，CPU缓存友好
        2. 前缀直接映射到节点索引，O(1)查找
        3. 子节点使用列表存储，避免 dict 开销
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._build_flat_trie()

    def _build_flat_trie(self):
        """构建展平的 Trie 结构"""
        logger.info("构建展平 Trie 结构...")

        # 节点结构: [children_start_idx, children_count, is_end]
        self._flat_nodes = []
        self._flat_children = []  # 所有子节点索引平铺存储

        # 前缀到节点索引的映射
        self._prefix_to_node_idx: Dict[Tuple[int, ...], int] = {}

        def build_node(prefix: Tuple[int, ...]) -> int:
            """递归构建节点，返回节点索引"""
            if prefix in self._prefix_to_node_idx:
                return self._prefix_to_node_idx[prefix]

            node_idx = len(self._flat_nodes)
            self._prefix_to_node_idx[prefix] = node_idx

            # 获取有效后续
            valid_next = self.trie.get_valid_next_tokens(list(prefix))

            children_start = len(self._flat_children)
            for child_idx in valid_next:
                self._flat_children.append(child_idx)

            is_end = len(prefix) == self.trie.num_levels
            self._flat_nodes.append([children_start, len(valid_next), is_end])

            # 递归构建子节点
            for child_idx in valid_next:
                build_node(prefix + (child_idx,))

            return node_idx

        # 从根节点开始构建
        build_node(tuple())

        logger.info(f"展平 Trie 完成: {len(self._flat_nodes)} 个节点")

    def _get_valid_next_indices_fast(self, prefix: List[int]) -> List[int]:
        """快速获取有效后续 token"""
        prefix_key = tuple(prefix)
        node_idx = self._prefix_to_node_idx.get(prefix_key)

        if node_idx is None:
            return []

        children_start, children_count, _ = self._flat_nodes[node_idx]
        return self._flat_children[children_start:children_start + children_count]

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        batch_size = input_ids.shape[0]
        vocab_size = scores.shape[1]
        device = scores.device

        for batch_idx in range(batch_size):
            current_sid_tokens = self._parse_generated_sid_tokens(input_ids[batch_idx])
            current_level = self._get_current_level(current_sid_tokens)

            if current_level >= self.trie.num_levels:
                if self.force_constraint:
                    mask = torch.full((vocab_size,), float("-inf"), device=device)
                    if self.eos_token_id is not None:
                        mask[self.eos_token_id] = 0
                    if self.sep_token_id is not None:
                        mask[self.sep_token_id] = 0
                    scores[batch_idx] = scores[batch_idx] + mask
                continue

            # 使用展平 Trie 查询
            valid_next_indices = self._get_valid_next_indices_fast(current_sid_tokens)

            if not valid_next_indices:
                continue

            valid_token_ids = []
            for idx in valid_next_indices:
                sid_token_str = f"<{self.trie.level_prefixes[current_level]}_{idx}>"
                if sid_token_str in self._sid_token_to_id:
                    valid_token_ids.append(self._sid_token_to_id[sid_token_str])

            if self.force_constraint:
                mask = torch.full((vocab_size,), float("-inf"), device=device)
                for token_id in valid_token_ids:
                    if 0 <= token_id < vocab_size:
                        mask[token_id] = 0
                scores[batch_idx] = scores[batch_idx] + mask

        return scores


# ==============================================================================
# Level 2: PrecomputedMaskProcessor - 预计算所有 Mask
# ==============================================================================

class PrecomputedMaskProcessor(SIDConstrainedLogitsProcessor):
    """
    Level 2 优化: 预计算所有前缀对应的 Mask

    优化点:
        1. 离线计算所有可能前缀的 mask
        2. 运行时直接查表，零计算
        3. 内存换时间策略

    内存占用估算:
        - 假设 100万 SID，5层，平均前缀数 ~250万
        - vocab_size=50000，每个 mask 100KB (float16)
        - 总内存: ~250GB (太大，需要限制)

    因此采用部分预计算策略：
        - 只预计算前 2 层的 mask（数量可控）
        - 深层使用 Trie 查询
    """

    def __init__(self, *args, precompute_levels: int = 2, **kwargs):
        """
        Args:
            precompute_levels: 预计算的层数（建议 2-3）
        """
        super().__init__(*args, **kwargs)
        self.precompute_levels = precompute_levels
        self._precompute_masks()

    def _precompute_masks(self):
        """预计算前 N 层的所有 mask"""
        logger.info(f"预计算前 {self.precompute_levels} 层的 mask...")

        vocab_size = len(self.tokenizer)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._precomputed_masks: Dict[Tuple[int, ...], torch.Tensor] = {}

        def build_masks_for_prefix(prefix: Tuple[int, ...]):
            """为指定前缀构建 mask"""
            if len(prefix) >= self.precompute_levels:
                return

            valid_next = self.trie.get_valid_next_tokens(list(prefix))

            if valid_next:
                mask = torch.full((vocab_size,), float("-inf"), device=device, dtype=torch.float16)
                level = len(prefix)

                for idx in valid_next:
                    sid_token_str = f"<{self.trie.level_prefixes[level]}_{idx}>"
                    if sid_token_str in self._sid_token_to_id:
                        token_id = self._sid_token_to_id[sid_token_str]
                        if 0 <= token_id < vocab_size:
                            mask[token_id] = 0

                self._precomputed_masks[prefix] = mask

                # 递归构建子前缀
                for idx in valid_next:
                    build_masks_for_prefix(prefix + (idx,))

        build_masks_for_prefix(tuple())

        self._mask_cache_hits = 0
        self._mask_cache_misses = 0

        logger.info(f"预计算 mask 完成: {len(self._precomputed_masks)} 个前缀")

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        batch_size = input_ids.shape[0]
        vocab_size = scores.shape[1]
        device = scores.device

        for batch_idx in range(batch_size):
            current_sid_tokens = self._parse_generated_sid_tokens(input_ids[batch_idx])
            current_level = self._get_current_level(current_sid_tokens)
            prefix_key = tuple(current_sid_tokens)

            if current_level >= self.trie.num_levels:
                if self.force_constraint:
                    mask = torch.full((vocab_size,), float("-inf"), device=device)
                    if self.eos_token_id is not None:
                        mask[self.eos_token_id] = 0
                    if self.sep_token_id is not None:
                        mask[self.sep_token_id] = 0
                    scores[batch_idx] = scores[batch_idx] + mask
                continue

            # 尝试使用预计算 mask（检查 vocab_size 匹配）
            if (len(current_sid_tokens) < self.precompute_levels
                and prefix_key in self._precomputed_masks):
                precomputed_mask = self._precomputed_masks[prefix_key]
                # 如果 vocab_size 不匹配，截取或填充
                if precomputed_mask.shape[0] != vocab_size:
                    if precomputed_mask.shape[0] > vocab_size:
                        mask = precomputed_mask[:vocab_size].to(device=device, dtype=scores.dtype)
                    else:
                        # 扩展 mask
                        mask = torch.full((vocab_size,), float("-inf"),
                                        device=device, dtype=scores.dtype)
                        mask[:precomputed_mask.shape[0]] = precomputed_mask.to(
                            device=device, dtype=scores.dtype)
                else:
                    mask = precomputed_mask.to(device=device, dtype=scores.dtype)
                scores[batch_idx] = scores[batch_idx] + mask
                self._mask_cache_hits += 1
            else:
                # 回退到 Trie 查询
                self._mask_cache_misses += 1
                valid_next_indices = self.trie.get_valid_next_tokens(current_sid_tokens)

                if not valid_next_indices:
                    continue

                mask = torch.full((vocab_size,), float("-inf"), device=device)
                for idx in valid_next_indices:
                    level_prefix = self.trie.level_prefixes[current_level]
                    sid_token_str = f"<{level_prefix}_{idx}>"
                    if sid_token_str in self._sid_token_to_id:
                        token_id = self._sid_token_to_id[sid_token_str]
                        if 0 <= token_id < vocab_size:
                            mask[token_id] = 0
                scores[batch_idx] = scores[batch_idx] + mask

        return scores

    def get_precompute_stats(self) -> Dict[str, Any]:
        """获取预计算统计"""
        total = self._mask_cache_hits + self._mask_cache_misses
        return {
            "precomputed_masks": len(self._precomputed_masks),
            "cache_hits": self._mask_cache_hits,
            "cache_misses": self._mask_cache_misses,
            "hit_rate": self._mask_cache_hits / total if total > 0 else 0,
        }


# ==============================================================================
# Level 3: OptimizedBatchProcessor - 批量处理优化
# ==============================================================================

class OptimizedBatchProcessor(SIDConstrainedLogitsProcessor):
    """
    Level 3 优化: 向量化批量处理

    优化点:
        1. 使用 torch 向量化操作替代 Python 循环
        2. 批量解析 SID tokens
        3. 使用 scatter 操作构建 mask

    注意: 这个优化对 SID 约束场景效果有限，因为每个 batch 的前缀不同
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._precompute_all_token_ids()

    def _precompute_all_token_ids(self):
        """预计算所有层级和索引到 token_id 的映射"""
        max_level = self.trie.num_levels
        max_codebook = max(self.trie.codebook_sizes)

        # 创建 tensor: [level, idx] -> token_id
        self._token_id_map = torch.full(
            (max_level, max_codebook),
            -1,
            dtype=torch.long
        )

        for level in range(max_level):
            level_prefix = self.trie.level_prefixes[level]
            for idx in range(self.trie.codebook_sizes[level]):
                sid_token_str = f"<{level_prefix}_{idx}>"
                if sid_token_str in self._sid_token_to_id:
                    self._token_id_map[level, idx] = self._sid_token_to_id[sid_token_str]

        # 移动到 GPU
        if torch.cuda.is_available():
            self._token_id_map = self._token_id_map.cuda()

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        batch_size = input_ids.shape[0]
        vocab_size = scores.shape[1]
        device = scores.device

        # 为每个 batch 单独处理（SID 约束的特殊性）
        for batch_idx in range(batch_size):
            current_sid_tokens = self._parse_generated_sid_tokens(input_ids[batch_idx])
            current_level = len(current_sid_tokens)

            if current_level >= self.trie.num_levels:
                if self.force_constraint:
                    mask = torch.full((vocab_size,), float("-inf"), device=device)
                    if self.eos_token_id is not None:
                        mask[self.eos_token_id] = 0
                    if self.sep_token_id is not None:
                        mask[self.sep_token_id] = 0
                    scores[batch_idx] = scores[batch_idx] + mask
                continue

            valid_next_indices = self.trie.get_valid_next_tokens(current_sid_tokens)

            if not valid_next_indices:
                continue

            # 向量化获取 token_ids
            valid_indices_tensor = torch.tensor(valid_next_indices, device=device)
            valid_token_ids = self._token_id_map[current_level, valid_indices_tensor]
            valid_token_ids = valid_token_ids[valid_token_ids >= 0]

            if self.force_constraint and len(valid_token_ids) > 0:
                mask = torch.full((vocab_size,), float("-inf"), device=device)
                mask[valid_token_ids] = 0
                scores[batch_idx] = scores[batch_idx] + mask

        return scores


# ==============================================================================
# Level 4: HybridProcessor - 混合优化策略（推荐）
# ==============================================================================

class HybridProcessor(PrecomputedMaskProcessor):
    """
    Level 4: 混合优化 - 结合所有优化策略

    组合策略:
        1. 预计算前 2 层 mask（Level 2）
        2. 深层使用展平 Trie（Level 1）
        3. 向量化 token 转换（Level 3）
    """

    def __init__(self, *args, precompute_levels: int = 2, **kwargs):
        # 先调用父类初始化预计算 mask
        super().__init__(*args, precompute_levels=precompute_levels, **kwargs)

        # 再构建展平 Trie 用于深层
        self._build_flat_trie_for_deep()

        # 预计算 token_id 映射（来自 Level 3）
        self._precompute_token_id_map()

    def _precompute_token_id_map(self):
        """预计算所有层级和索引到 token_id 的映射"""
        max_level = self.trie.num_levels
        max_codebook = max(self.trie.codebook_sizes)

        # 创建 tensor: [level, idx] -> token_id
        self._token_id_map = torch.full(
            (max_level, max_codebook),
            -1,
            dtype=torch.long
        )

        for level in range(max_level):
            level_prefix = self.trie.level_prefixes[level]
            for idx in range(self.trie.codebook_sizes[level]):
                sid_token_str = f"<{level_prefix}_{idx}>"
                if sid_token_str in self._sid_token_to_id:
                    self._token_id_map[level, idx] = self._sid_token_to_id[sid_token_str]

        # 移动到 GPU
        if torch.cuda.is_available():
            self._token_id_map = self._token_id_map.cuda()

    def _build_flat_trie_for_deep(self):
        """为深层构建展平 Trie"""
        logger.info("为深层构建展平 Trie...")

        self._deep_flat_nodes = {}

        # 只构建深层的前缀（第 2 层开始）
        for sid in self.trie.get_all_sids():
            for depth in range(self.precompute_levels, self.trie.num_levels):
                prefix = tuple(sid[:depth])
                if prefix in self._deep_flat_nodes:
                    continue

                valid_next = self.trie.get_valid_next_tokens(list(prefix))
                self._deep_flat_nodes[prefix] = valid_next

        logger.info(f"深层 Trie 节点数: {len(self._deep_flat_nodes)}")

    def _get_valid_next_fast(self, prefix: List[int]) -> List[int]:
        """快速获取有效后续"""
        prefix_key = tuple(prefix)

        if len(prefix) < self.precompute_levels:
            # 使用父类方法（预计算）
            return self.trie.get_valid_next_tokens(prefix)
        else:
            # 使用展平 Trie
            return self._deep_flat_nodes.get(prefix_key, [])

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        batch_size = input_ids.shape[0]
        vocab_size = scores.shape[1]
        device = scores.device

        for batch_idx in range(batch_size):
            current_sid_tokens = self._parse_generated_sid_tokens(input_ids[batch_idx])
            current_level = len(current_sid_tokens)
            prefix_key = tuple(current_sid_tokens)

            if current_level >= self.trie.num_levels:
                if self.force_constraint:
                    mask = torch.full((vocab_size,), float("-inf"), device=device)
                    if self.eos_token_id is not None:
                        mask[self.eos_token_id] = 0
                    if self.sep_token_id is not None:
                        mask[self.sep_token_id] = 0
                    scores[batch_idx] = scores[batch_idx] + mask
                continue

            # Level 2: 预计算 mask（浅层）
            if current_level < self.precompute_levels and prefix_key in self._precomputed_masks:
                precomputed_mask = self._precomputed_masks[prefix_key]
                # 如果 vocab_size 不匹配，截取或填充
                if precomputed_mask.shape[0] != vocab_size:
                    if precomputed_mask.shape[0] > vocab_size:
                        mask = precomputed_mask[:vocab_size].to(device=device, dtype=scores.dtype)
                    else:
                        # 扩展 mask
                        mask = torch.full((vocab_size,), float("-inf"),
                                        device=device, dtype=scores.dtype)
                        mask[:precomputed_mask.shape[0]] = precomputed_mask.to(
                            device=device, dtype=scores.dtype)
                else:
                    mask = precomputed_mask.to(device=device, dtype=scores.dtype)
                scores[batch_idx] = scores[batch_idx] + mask
                self._mask_cache_hits += 1
                continue

            # Level 1 + 3: 展平 Trie + 向量化
            self._mask_cache_misses += 1
            valid_next_indices = self._get_valid_next_fast(current_sid_tokens)

            if not valid_next_indices:
                continue

            # 向量化转换
            valid_indices_tensor = torch.tensor(valid_next_indices, device=device)
            valid_token_ids = self._token_id_map[current_level, valid_indices_tensor]
            valid_token_ids = valid_token_ids[valid_token_ids >= 0]

            if self.force_constraint and len(valid_token_ids) > 0:
                mask = torch.full((vocab_size,), float("-inf"), device=device)
                mask[valid_token_ids] = 0
                scores[batch_idx] = scores[batch_idx] + mask

        return scores


# ==============================================================================
# 工具函数：对比评估
# ==============================================================================

PROCESSOR_REGISTRY = {
    "original": SIDConstrainedLogitsProcessor,
    "cached": CachedSIDConstrainedLogitsProcessor,
    "flat": FlatTrieProcessor,
    "precomputed": PrecomputedMaskProcessor,
    "batch": OptimizedBatchProcessor,
    "hybrid": HybridProcessor,
}

def create_processor(processor_type: str, trie: SIDTrie, tokenizer: Any, **kwargs):
    """
    工厂函数：创建指定类型的 processor

    Args:
        processor_type: 类型名称 (original/cached/flat/precomputed/batch/hybrid)
        trie: SIDTrie 实例
        tokenizer: Tokenizer 实例
        **kwargs: 额外参数

    Returns:
        SIDConstrainedLogitsProcessor 实例
    """
    if processor_type not in PROCESSOR_REGISTRY:
        raise ValueError(f"Unknown processor type: {processor_type}. "
                        f"Available: {list(PROCESSOR_REGISTRY.keys())}")

    processor_class = PROCESSOR_REGISTRY[processor_type]
    return processor_class(trie=trie, tokenizer=tokenizer, **kwargs)

    