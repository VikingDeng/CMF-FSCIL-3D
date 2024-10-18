from src.dataset import ShapeNetTest
from tqdm import tqdm
data1 = ShapeNetTest(task_id=6)
data2 = ShapeNetTest(task_id=6)
assert len(data1) == len(data2)
for i in tqdm(range(len(data1))):
    _,target1,_ = data1[i]
    _,target2,_ = data1[i]
    assert target1 == target2