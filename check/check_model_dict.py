import torch
model_path = "outputs/best_checkpoint.pth"
ck = torch.load(model_path)
model_state_dict = ck['model_state_dict']
res = []
for n,p in model_state_dict.items():
    res.append(n)
with open("tmp.txt","w") as f:
    f.writelines(res)