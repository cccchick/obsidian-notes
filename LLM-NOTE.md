LLM笔记

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![UV](https://img.shields.io/badge/Package-UV-green)](https://github.com/astral-sh/uv)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
---

##  环境配置（UV 版）

```bash
# 1. 安装 uv（已装可跳过）
pip install uv

# 2. 创建虚拟环境并激活
uv venv llm-env
source llm-env/bin/activate  # Linux/Mac
# llm-env\Scripts\activate   # Windows

# 3. 安装依赖（课程所需）
uv pip install jupyterlab wandb transformers datasets tiktoken

# 4. 启动 JupyterLab
jupyter lab
```
---
## 使用W&B记录
```bash
# 没有网络环境时使用离线模式
wandb.init(mode="offline", project="diy-llm")
```
---
#  分词器
---
## 一、核心认知

Tokenizer 是 LLM的**独立组件**，但**强耦合**关系：
- 独立训练：有自己的语料和训练流程
- 强耦合：LLM 必须用同一个 Tokenizer，不能换
- 通过正则预处理 + 统计方法构建词表（vocab），建立**文本片段 ↔ token ID**的双向映射
---
## 二、训练流程

- 准备语料 → 初始化基础单元 → 统计并迭代合并 → 输出产物与编码解码
### 2.1准备语料
- 覆盖目标场景的多样化文本（不同体裁、多语言）
- 完成文本清洗：去重、乱码修正、统一 UTF-8 编码、去除无关元数据
- 敏感信息脱敏：合规要求，同时降低语料噪声，提升分词统计效率
- 多语言场景下，需避免高资源语言主导词表，导致低资源语言 token 碎片化。
[[文本脱敏]]
### 2.2预分词
预分词的核心是将原始文本切分为可统计、可合并的基础单元

#### 1.基于空格和标点的切分
```python
import re 
def part(text):
# 将标点符号单独拆开，按空格分割
    text = re.sub(r'([.,!?;:()"\'\[\]{}])', r' \1 ', text)
    tokens = text.split()
    return tokens
```
#### 2.基于 Unicode 类别划分
```python
import unicodedata

def get_char_category(ch: str) -> str:
    """获取字符的Unicode类别"""
    cat = unicodedata.category(ch)
    if '\u4e00' <= ch <= '\u9fff':
        return "CJK"
    if ch.isdigit():
        return "DIGIT"
    if ch.isalpha():
        return "ALPHA"
    if cat.startswith("P"):
        return "PUNCT"
    return "OTHER"

def segment_by_unicode_category(text: str):
    """按Unicode类别切分文本"""
    if not text:
        return []
    segments = []
    buffer = [text[0]]
    prev_type = get_char_category(text[0])

    for ch in text[1:]:
        curr_type = get_char_category(ch)
        if curr_type == prev_type:
            buffer.append(ch)
        else:
            segments.append(("".join(buffer), prev_type))
            buffer = [ch]
            prev_type = curr_type
    segments.append(("".join(buffer), prev_type))

    return [seg for seg, _ in segments]


```
#### 3.字节级切分
```python
def tokenize_byte_level(text):
    """字节级分词，返回UTF-8字节的十六进制列表"""
    tokens = []
    for ch in text:
        utf8_bytes = ch.encode("utf-8")
        hex_bytes = [f"{b:02X}" for b in utf8_bytes]
        print(f"{ch} 转化为UTF-8字节序列：{hex_bytes}")
        tokens.extend(hex_bytes)
    return tokens

# 测试
if __name__ == "__main__":
    s = "All for learners！"
    print(tokenize_byte_level(s))
```
### 3.统计并迭代合并

| 算法            | 核心原理                            | 适用场景           |
| ------------- | ------------------------------- | -------------- |
| BPE           | 贪心合并频次最高的相邻 token 对             | 大规模语料、高频子词压缩   |
| WordPiece     | 优先合并能最大化提升语料似然的子词对              | MLM 类模型、控制 OOV |
| Unigram LM    | 从大词表出发，迭代剪枝低概率 token，最大化语料似然    | 多语言场景、低频词友好    |
| SentencePiece | 语言无关的分词框架，内置 BPE/Unigram，支持字节兜底 | 多语种、端到端训练      |
### 4.输出产物与编码解码
训练完成后，核心输出 2 个文件： 
 - vocab文件：记录所有 token 与对应 ID 的映射，是编码解码的核心索引 
- merges文件：按顺序记录子词合并规则 / 概率模型，保证编码可逆
- 后续需完成：验证分词效果（平均 token 数、碎片化程度、跨语言平衡度）、版本管理，确保训练与推理阶段分词器版本一致。
---
## 三、分词器

### 1.字符分词器
- 将文本拆解为单个字符（英文字母、中文单字），直接映射为 Unicode 编码
- 优点：词表极小、无 OOV 问题
- 缺点：序列过长、显存消耗大、语义稀疏
- ```python
  class CharacterTokenizer:
    def __init__(self):
        pass

    def encode(self, text):
        """文本→Unicode编码索引列表"""
        return [ord(ch) for ch in text]

    def decode(self, indices):
        """索引列表→原始文本"""
        return ''.join([chr(i) for i in indices])

# 测试
if __name__ == "__main__":
    tokenizer = CharacterTokenizer()
    string = "hi，很好的，chick！🐋"
    # 编码
    indices = tokenizer.encode(string)
    print("编码ID:", indices)
    # 解码
    decoded = tokenizer.decode(indices)
    print("解码结果:", decoded)
    # 验证可逆
    assert string == decoded, "编码解码不一致!"
  ```
### 2.字节分词器
- 直接对 UTF-8 二进制字节操作，基础词表固定为 256（0x00-0xFF），彻底解决 OOV 问题
- 优点：无 OOV、兼容所有字符、稳定性极强
- 缺点：无压缩能力，序列长度最长
- ```python
  class ByteTokenizer:
    def __init__(self):
        self.vocab_size = 256

    def encode(self, text: str):
        """文本→UTF-8字节列表"""
        return list(text.encode("utf-8"))

    def decode(self, indices):
        """字节列表→原始文本"""
        return bytes(indices).decode("utf-8")

# 测试
if __name__ == "__main__":
    tokenizer = ByteTokenizer()
    text = "Hello, 🌍! 你好!"
    encoded = tokenizer.encode(text)
    print("编码字节:", encoded)
    decoded = tokenizer.decode(encoded)
    print("解码结果:", decoded)
  ```
### 3.词级分词器
- 基于正则 / 分词算法将文本切分为独立语义的词，为每个词分配唯一 ID
- 优点：token 保留完整语义、序列长度短
- 缺点：词表爆炸、OOV 问题严重
- ```python
  import regex

# 预分词正则表达式
TOKENIZER_REGEX = r"\p{L}+|\p{N}+|[^\p{L}\p{N}\s]+|\s+"

class WordTokenizer:
    def __init__(self, pattern=TOKENIZER_REGEX):
        self.pattern = pattern
        self.word2id = {}
        self.id2word = {}

    def build_vocab(self, texts):
        """基于训练文本构建词表"""
        vocab = set()
        for text in texts:
            segments = regex.findall(self.pattern, text)
            vocab.update(segments)
        vocab = sorted(vocab)
        self.word2id = {w: i for i, w in enumerate(vocab)}
        self.id2word = {i: w for w, i in self.word2id.items()}

    def encode(self, text):
        """文本→token ID列表，未登录词为-1"""
        segments = regex.findall(self.pattern, text)
        return [self.word2id.get(seg, -1) for seg in segments]

    def decode(self, ids):
        """ID列表→原始文本"""
        return "".join(self.id2word.get(i, "<UNK>") for i in ids)

# 测试
if __name__ == "__main__":
    train_texts = ["这只猫🐈很可爱", "the quick brown fox jumps over the lazy 🐕‍🦺"]
    tokenizer = WordTokenizer()
    tokenizer.build_vocab(train_texts)
    print("词表大小:", len(tokenizer.word2id))
    
    test_text = "敏捷的棕色狐狸🦊"
    encoded = tokenizer.encode(test_text)
    print("编码ID:", encoded)
    decoded = tokenizer.decode(encoded)
    print("解码结果:", decoded)
  ```
### 4.BPE分词器
- 在字符级与词级之间找平衡，统计相邻 token 对的频次，迭代合并频次最高的对，直到达到预设词表大小
- 优点：词表大小适中、OOV 极少、压缩率高、序列长度适中
  缺点：训练需迭代统计，对低频词效果一般
- [[BPE]]
---
## 四、DeepSeek 分词器(工业级分词器

### 4.1 官方分词器加载
```python
from transformers import AutoTokenizer

# 加载DeepSeek Coder分词器
MODEL_NAME = "deepseek-ai/deepseek-coder-6.7b-instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print(f"词表大小: {len(tokenizer.get_vocab())}")

# 编码解码测试
text = "hello world!注意力机制是AI的核心技术。 🚀 🚀"
encoded_ids = tokenizer.encode(text, add_special_tokens=False)
tokens = tokenizer.convert_ids_to_tokens(encoded_ids)
decoded_text = tokenizer.decode(encoded_ids)

print(f"原文: {text}")
print(f"子词序列: {tokens}")
print(f"解码结果: {decoded_text}")
```
### 4.2 DeepSeek 分词器简易完整实现
```python
import regex as re
from collections import Counter
from typing import List, Tuple, Dict, Iterable
import json
import base64

# 配置：DeepSeek风格预分词正则
DEEPSEEK_REGEX = r"\p{L}+|\p{N}+|[^\p{L}\p{N}\s]+|\s+"

# 基础工具函数
def pretokenize(text:str):
    """按DeepSeek风格正则预分词"""
    return re.findall(DEEPSEEK_REGEX, text)

def bytes2tokens(b:bytes):
    """UTF-8字节→latin1字符，保证字节序列可逆"""
    return [bytes([x]).decode('latin1') for x in b]

def tokens2bytes(tokens):
    """latin1字符→原始UTF-8字节"""
    return b''.join([t.encode('latin1') for t in tokens])

# BPE训练核心函数
def build_corpus(texts):
    """构建字节级BPE语料"""
    corpus = []
    for text in texts:
        for chunk in pretokenize(text):
            corpus.append(bytes2tokens(chunk.encode('utf-8')))
    return corpus

def pair_freq(corpus: List[List[str]]):
    """统计相邻token对频次"""
    pairs = Counter()
    for word in corpus:
        for i in range(len(word)-1):
            pairs[(word[i], word[i+1])] += 1
    return pairs

def merge_pair(word: List[str], pair: Tuple[str,str]):
    """合并指定token对"""
    a, b = pair
    merged = []
    i = 0
    while i < len(word):
        if i < len(word)-1 and word[i]==a and word[i+1]==b:
            merged.append(a+b)
            i += 2
        else:
            merged.append(word[i])
            i += 1
    return merged

def train_bpe(texts: Iterable[str], vocab_size: int=5000, num_merges: int=None) -> Tuple[List[Tuple[str,str]], List[str]]:
    """训练字节级BPE模型"""
    corpus = build_corpus(texts)
    base_tokens = [bytes([i]).decode('latin1') for i in range(256)]
    merges: List[Tuple[str,str]] = []
    merged_set = set()
    cur_vocab_size = 256

    merge_steps = num_merges or (vocab_size - 256)

    for _ in range(merge_steps):
        pfreq = pair_freq(corpus)
        if not pfreq:
            break
        best_pair, _ = pfreq.most_common(1)[0]
        if cur_vocab_size + 1 > vocab_size:
            break
        merges.append(best_pair)
        corpus = [merge_pair(word, best_pair) for word in corpus]
        merged_set.add(best_pair[0]+best_pair[1])
        cur_vocab_size += 1

    # 追加特殊token
    special_tokens = ["<pad>", "<bos>", "<eos>", "<unk>"]
    vocab_tokens = special_tokens + base_tokens + sorted(merged_set)
    return merges, vocab_tokens

# DeepSeek V3 Tokenizer主类
class DeepSeekV3Tokenizer:
    def __init__(self, merges: List[Tuple[str,str]], vocab_tokens: List[str]):
        self.merges = merges
        self.vocab_tokens = vocab_tokens
        self.token2id = {tok:i for i, tok in enumerate(vocab_tokens)}
        self.id2token = {i:tok for tok,i in self.token2id.items()}
        self.ranks = {pair:i for i,pair in enumerate(merges)}
        # 特殊token定义
        self.pad_token = "<pad>"
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"
        self.unk_token = "<unk>"

    def encode_chunk(self, chunk: str) -> List[str]:
        """对单个预分词块做BPE编码"""
        tokens = bytes2tokens(chunk.encode('utf-8'))
        # 应用合并规则
        for pair in self.merges:
            new_tokens = []
            i = 0
            a,b = pair
            while i < len(tokens):
                if i<len(tokens)-1 and tokens[i]==a and tokens[i+1]==b:
                    new_tokens.append(a+b)
                    i+=2
                else:
                    new_tokens.append(tokens[i])
                    i+=1
            tokens = new_tokens
        # OOV处理
        out = []
        for t in tokens:
            if t in self.token2id:
                out.append(t)
            else:
                out.extend([ch if ch in self.token2id else self.unk_token for ch in t])
        return out

    def encode(self, text: str, add_bos=False, add_eos=False, print_chunks=False):
        """完整文本编码入口"""
        ids = []
        if add_bos:
            ids.append(self.token2id[self.bos_token])
            if print_chunks: print(f"[Special] <bos> -> {self.token2id[self.bos_token]}")

        for chunk in pretokenize(text):
            toks = self.encode_chunk(chunk)
            chunk_ids = [self.token2id.get(t, self.token2id[self.unk_token]) for t in toks]
            if print_chunks:
                readable = []
                for t in toks:
                    try:
                        r = tokens2bytes([t]).decode('utf-8', errors='ignore')
                        readable.append(r if r else t.encode('latin1').hex())
                    except:
                        readable.append(t.encode('latin1').hex())
                print(f"[Chunk] \"{chunk}\" -> {readable} -> IDs: {chunk_ids}")
            ids.extend(chunk_ids)

        if add_eos:
            ids.append(self.token2id[self.eos_token])
            if print_chunks: print(f"[Special] <eos> -> {self.token2id[self.eos_token]}")
        return ids

    def decode(self, ids: Iterable[int]):
        """ID序列还原为文本"""
        byte_seq = bytearray()
        for i in ids:
            tok = self.id2token.get(i, self.unk_token)
            if tok in {self.pad_token, self.bos_token, self.eos_token}:
                continue
            byte_seq.extend(tokens2bytes(list(tok)))
        return byte_seq.decode('utf-8', errors='replace')

    def save(self, vocab_path: str, merges_path: str):
        """保存词表与合并规则"""
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(self.token2id, f, ensure_ascii=False, indent=2)
        merges_b64 = []
        for a, b in self.merges:
            a_bytes = a.encode('latin1')
            b_bytes = b.encode('latin1')
            merges_b64.append((
                base64.b64encode(a_bytes).decode('ascii'),
                base64.b64encode(b_bytes).decode('ascii')
            ))
        with open(merges_path, 'w', encoding='utf-8') as f:
            json.dump(merges_b64, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, vocab_path: str, merges_path: str):
        """加载已保存的分词器"""
        with open(vocab_path, 'r', encoding='utf-8') as f:
            token2id = json.load(f)
        vocab_tokens = [None] * (max(token2id.values()) + 1)
        for tok, idx in token2id.items():
            vocab_tokens[idx] = tok
        with open(merges_path, 'r', encoding='utf-8') as f:
            merges_b64 = json.load(f)
        merges = []
        for a_b64, b_b64 in merges_b64:
            a = base64.b64decode(a_b64).decode('latin1')
            b = base64.b64decode(b_b64).decode('latin1')
            merges.append((a, b))
        return cls(merges, vocab_tokens)

# 训练入口函数
def train_tokenizer(texts, vocab_size=5000, num_merges=None):
    merges, vocab_tokens = train_bpe(texts, vocab_size=vocab_size, num_merges=num_merges)
    return DeepSeekV3Tokenizer(merges, vocab_tokens)

# 测试示例
if __name__ == "__main__":
    train_texts = [
        "Transformer是AI的核心技术。",
        "DeepSeek分词器支持中文、英文、emoji等多语言。",
        "Hello, 世界! 🌍🚀",
    ]
    # 训练分词器
    tokenizer = train_tokenizer(train_texts, vocab_size=1024)
    print(f"词表大小: {len(tokenizer.vocab_tokens)}")
    # 编码测试
    txt = "注意力机制是AI的核心技术。 🚀 🚀"
    ids = tokenizer.encode(txt, add_bos=True, add_eos=True, print_chunks=True)
    print("Token ID:", ids)
    # 解码测试
    decoded = tokenizer.decode(ids)
    print("解码结果:", decoded)
    print("编码解码可逆:", decoded == txt)
```
### 总结
- latin1 编码的作用：单字节编码，可将 0-255 的任意字节映射为 Unicode 字符，保证多字节字符拆分后不丢失信息，实现编码完全可逆
- 字节级 BPE：以 UTF-8 字节为基础单元，彻底解决 OOV 问题，适配所有语言与特殊字符
- 正则预分词：按字符类型切分，对中文、代码、emoji 都有良好的适配性

