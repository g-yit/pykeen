from src.pykeen.triples.triples_factory import _get_triple_mask

import torch
from typing import Collection, Union, Optional

# 示例数据
ids = {1, 2}
triples = torch.tensor([[1, 2, 3], [4, 5, 6], [1, 5, 2], [2, 2, 2]])
columns = [0, 2]
invert = False

# 调用函数
mask = _get_triple_mask(ids, triples, columns, invert)
print(mask)
