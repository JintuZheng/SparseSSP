import torch

def repmode_task_embedding_func(task_id, num_tasks):
    N = task_id.shape[0]
    task_embedding = torch.zeros((N, num_tasks), device = task_id.device)
    for i in range(N):
        task_embedding[i, task_id[i]] = 1
    return task_embedding