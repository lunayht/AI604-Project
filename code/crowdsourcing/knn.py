for ex, lb in zip(features, labels):
        sim = torch.mm(ex.unsqueeze(0), train_features)
        nbr, ind = sim.topk(k=K, dim=-1)
        nbr_lab  = torch.gather(train_labels.expand(ex.shape[0], -1), index=ind, dim=-1)
        unique, lab_cnt  = nbr_lab.unique(return_counts=True)
        if lb == (unique[torch.argmax(lab_cnt)]):
            num_correct += 1